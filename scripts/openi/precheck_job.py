"""Run scoped V2 phases, enforce measured budgets and audit publishable artifacts."""
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import torch

from baselines.fedmd_kd import uniform_update
from core.data import read_train_sources, setup_data
from core.runtime import STEP_DEADLINE, batch_at, file_hash, init_client, initialize, normalize
from core.splits import digest
from scripts.openi.precheck_protocol import ValidationPanel, training_config, weighted_ce
from scripts.openi.precheck_science import base_models, read_rows, run_frequency, run_heterogeneity, run_work1
from scripts.openi.preflight import inspect
from scripts.openi.state import read_json, source_identity, write_json


MODES = {
    "env_check": (), "work1_precheck": ("work1",), "work2_heterogeneity": ("work2_heterogeneity",),
    "work2_frequency": ("work2_frequency",),
    "precheck_all": ("work1", "work2_heterogeneity", "work2_frequency"),
}
PACKAGES = ("torch", "torchvision", "scipy", "numpy", "scikit-learn", "PyYAML", "Pillow", "matplotlib")
TABLES = {
    "env_check": ("environment.json", "budget_estimate.json", "splits_manifest.json"),
    "base_2x2": ("checkpoints/base_2x2/quality.json", "checkpoints/base_2x2/snapshots.json",
                 "checkpoints/base_2x2/metrics/rounds.csv", "checkpoints/base_2x2/checkpoints/manifest.json"),
    "work1": tuple("work1/"+p for p in ("clean_mixed_comparison.csv", "loss_random_comparison.csv",
             "training_curves.csv", "sample_signals.csv", "paired_interventions.csv",
             "signal_value_correlations.csv", "corruption_detection.csv")),
    "work2_heterogeneity": tuple("work2_heterogeneity/"+p for p in ("heterogeneity_main_results.csv",
                       "kd_collaboration_gains.csv", "client_level_results.csv")),
    "work2_frequency": tuple("work2_frequency/"+p for p in ("teacher_student_metadata.csv",
                       "frequency_views.png", "delta_r_matrix.csv", "unlabeled_reliability_scores.csv",
                       "score_delta_r_correlations.csv", "runtime_statistics.json")),
}


def inspect_environment(device, paths):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10+ is required")
    if device.type == "cuda":
        report = inspect(paths)
    else:
        report = {"runtime": {"os": platform.system(), "python": platform.python_version(),
                              "gpu": None, "cuda_build": torch.version.cuda},
                  "cuda_kernel_verified": False,
                  "disk": {str(p): dict(zip(("total_bytes", "used_bytes", "free_bytes"), shutil.disk_usage(p)))
                           for p in paths}}
    report["packages"] = {name: version(name) for name in PACKAGES}
    report["device"] = str(device)
    return report


def budget_plan(cfg, modes, *, include_base=None):
    n = len(cfg["clients"])
    local = n*cfg["rounds"]*cfg["local_steps"]
    kd = n*cfg["rounds"]*cfg["kd_steps"]
    if include_base is None:
        include_base = bool(modes)
    return {
        "local_ce_updates": local*(include_base+3*("work1" in modes)+10*("work2_heterogeneity" in modes)),
        "standard_kd_updates": kd*5*("work2_heterogeneity" in modes),
        "work1_intervention_upper_updates": (
            n*2*6*cfg["probe1"]["quantile_bins"]*cfg["work1"]["repeats"]*5*cfg["work1"]["steps"]
            if "work1" in modes else 0),
        "frequency_updates": (n*(n-1)*3*cfg["probe2"]["repeats"]*len(cfg["probe2"]["domains"])*cfg["probe2"]["delta_steps"]
                              if "work2_frequency" in modes else 0),
    }


def timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, perf_counter()-start


def benchmark(cfg, data, panel, device, plan):
    clients = [init_client(c["model"], cfg, device, i) for i, c in enumerate(cfg["clients"])]
    batch_size = cfg["batch_size"]

    def load_batches():
        return [batch_at(dataset, range(batch_size), device)[0] for dataset in data.train]

    batches, data_seconds = timed(device, load_batches)
    forward = []
    with torch.inference_mode():
        for index, ((model, _), batch) in enumerate(zip(clients, batches, strict=True)):
            logits = model.eval()(normalize(batch))
            if logits.shape != (batch_size, 10) or not logits.isfinite().all().item():
                raise ValueError("Environment check got invalid CIFAR logits")
            forward.append({"client": index, "model": cfg["clients"][index]["model"], "shape": list(logits.shape)})
    short = {**cfg, "local_steps": 1, "kd_steps": 1}
    (local, metrics), local_time = timed(device, lambda: weighted_ce(short, data, device, "mixed_ce")(clients, 0))
    (kd, _), kd_time = timed(device, lambda: uniform_update(short, data, device)(clients, 0))
    _, eval_time = timed(device, lambda: panel.evaluate([m for m, _ in clients], data, cfg, device))
    model_and_momentum_bytes = sum(
        sum(t.numel()*t.element_size() for t in model.state_dict().values()) +
        sum(state["momentum_buffer"].numel()*state["momentum_buffer"].element_size()
            for state in opt.state.values() if "momentum_buffer" in state)
        for model, opt in clients)
    measured = {"sample_local_steps": local, "sample_local_wall_seconds": local_time,
                "sample_local_ce_loss": metrics["local_ce_loss"], "sample_kd_steps": kd,
                "sample_kd_wall_seconds": kd_time, "sample_validation_wall_seconds": eval_time,
                "sample_validation_images_per_client": len(data.val_clean[0]),
                "sample_group_state_tensor_bytes": model_and_momentum_bytes,
                "sample_forwards": forward,
                "sample_data_batch_images": batch_size*len(clients),
                "sample_data_wall_seconds": data_seconds,
                "sample_data_images_per_second": batch_size*len(clients)/data_seconds}
    return estimate_from_sample(cfg, plan, measured)


def estimate_from_sample(cfg, plan, measured):
    n = len(cfg["clients"])
    arms = plan["local_ce_updates"] // (n*cfg["rounds"]*cfg["local_steps"])
    quality_checks = arms
    work1_arms = plan["work1_intervention_upper_updates"]/cfg["work1"]["steps"]
    freq_arms = plan["frequency_updates"]/cfg["probe2"]["delta_steps"]
    eval_rounds = arms*cfg["rounds"] + quality_checks + (work1_arms+freq_arms)/n
    eval_rounds = max(1, eval_rounds)
    eval_scale = max(1, cfg["limits"]["val"]/measured["sample_validation_images_per_client"])
    local_rate = measured["sample_local_wall_seconds"]/measured["sample_local_steps"]
    kd_rate = measured["sample_kd_wall_seconds"]/measured["sample_kd_steps"]
    estimated = (local_rate*(plan["local_ce_updates"]+plan["work1_intervention_upper_updates"])
                 +kd_rate*(plan["standard_kd_updates"]+plan["frequency_updates"])
                 +measured["sample_validation_wall_seconds"]*eval_rounds*eval_scale)*cfg["budget"]["estimate_margin"]
    return {**{key: value for key, value in measured.items() if key.startswith("sample_")},
            "estimated_updates": plan,
            "estimated_compute_wall_seconds": estimated,
            "estimated_points_at_configured_rate": estimated/3600*cfg["budget"]["points_per_hour"],
            "estimated_checkpoint_storage_bytes": math.ceil(
                measured["sample_group_state_tensor_bytes"]*(2*arms+4)*1.1) if arms else 0,
            "configured_points_per_hour": cfg["budget"]["points_per_hour"],
            "excludes": ["queue", "archive upload", "unmeasured storage latency"],
            "purpose": "throughput_estimate"}


def affordable_prefix(cfg, pending, base_needed, measurement, available_seconds):
    for count in range(len(pending), -1, -1):
        stages = pending[:count]
        estimate = estimate_from_sample(cfg, budget_plan(cfg, stages, include_base=base_needed), measurement)
        if estimate["estimated_compute_wall_seconds"] <= available_seconds:
            return stages, estimate
    raise TimeoutError("Estimated base training does not fit the current task budget")


def csv_check(path):
    bad = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "-infinity", "+infinity"}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, strict=True)
        header = next(rows, [])
        if not header or len(set(header)) != len(header):
            raise ValueError(f"Invalid CSV header: {path}")
        count = 0
        for values in rows:
            if len(values) != len(header) or any(v.strip().casefold() in bad for v in values):
                raise ValueError(f"Malformed or non-finite CSV record: {path}")
            count += 1
    if not count:
        raise ValueError(f"CSV contains no records: {path}")
    return count


def bad_number(value):
    raise ValueError(f"Invalid JSON number: {value}")


def inspect_artifact(path):
    entry = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    if path.suffix == ".csv":
        entry["rows"] = csv_check(path)
    elif path.suffix == ".jsonl":
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not isinstance(json.loads(line, parse_constant=bad_number), dict):
                    raise ValueError(f"Invalid JSONL record: {path}")
                count += 1
        if not count:
            raise ValueError(f"JSONL contains no records: {path}")
        entry["rows"] = count
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_number)
    elif path.suffix == ".png":
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    return entry


def audited_files(root, errors=None):
    records = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file() or any(p.startswith(".") for p in relative.parts):
            continue
        if path.name in {"file_hashes.json", "upload_status.json"} or path.suffix == ".pending":
            continue
        name = relative.as_posix()
        try:
            records[name] = inspect_artifact(path)
        except (OSError, ValueError, csv.Error) as error:
            if errors is None:
                raise
            errors[name] = {"type": type(error).__name__, "message": str(error)}
    return records


def phase_receipt(root, stage, identity_hash):
    required = list(TABLES[stage])
    plots = {
        "work1": ("work1_training_curves.png", "work1_intervention_plot.png"),
        "work2_heterogeneity": ("heterogeneity_comparison.png",),
        "work2_frequency": ("delta_r_heatmap.png", "score_vs_gain.png"),
    }
    required.extend("figures/"+name for name in plots.get(stage, ()))
    if stage == "base_2x2":
        base = root / "checkpoints" / "base_2x2"
        source = read_json(base / "snapshots.json")["paths"]
        required.extend(source.values())
        checks = read_json(base / "checkpoints" / "manifest.json")["client_sha256"]
        for i, sha in enumerate(checks):
            name = f"checkpoints/base_2x2/checkpoints/client_{i}.pt"
            if file_hash(root / name) != sha:
                raise ValueError(f"Base checkpoint export {i} failed integrity")
            required.append(name)
    result = {}
    for name in required:
        p = root / name
        if p.suffix == ".csv":
            csv_check(p)
        result[name] = file_hash(p)
    return {"identity_sha256": identity_hash, "files": result}


def completed(root, stage, identity_hash):
    marker = root / "progress" / f"{stage}.json"
    if not marker.exists():
        return False
    if read_json(marker) != phase_receipt(root, stage, identity_hash):
        raise ValueError(f"Previously completed phase failed validation: {stage}")
    return True


def v2_job(cfg, root, data_root, inputs, source, upload_checkpoint, deployment=None):
    from scripts.openi.precheck_figures import draw
    from scripts.openi.precheck_reports import run_reports

    mode = cfg["mode"]
    stages = MODES[mode]
    base_cfg = training_config(cfg, root, data_root, "heterogeneous")
    if (not 0 < cfg["downweight"]["fraction"] <= 1 or not 0 < cfg["downweight"]["weight"] < 1
            or min(cfg["work1"]["steps"], cfg["work1"]["repeats"]) < 1):
        raise ValueError("Invalid intervention or matched weight budget")
    mix = cfg["mixture"]
    if not (0 < mix["moderate_fraction"] and 0 < mix["severe_fraction"]
            and mix["moderate_fraction"] + mix["severe_fraction"] <= 1
            and 1 <= mix["moderate"] < mix["severe"] <= 5):
        raise ValueError("Invalid corruption mixture")
    if not base_cfg["split"]["proxy100_size"] and (base_cfg["proxy"] == "cifar100_proxy" or
                                                  "cifar100_proxy" in base_cfg["probe2"]["domains"]):
        raise ValueError("CIFAR-100 proxy was requested but disabled")
    env_only = mode == "env_check"
    device = initialize(base_cfg)
    env = inspect_environment(device, (root, data_root))
    env["deployment"] = deployment or {"managed": False, "c2net_prepare": False}
    selected_runtime = {k: env[k] for k in ("runtime", "device", "packages")}
    stable = {k: v for k, v in base_cfg.items() if k not in
              {"mode", "run_id", "data_root", "out", "manifest", "checkpoint_dir", "budget", "dataset_subdir",
               "resume", "restore_subdir", "restore_mount", "git_commit"}}
    identity = {"config_sha256": digest(stable), "source_sha256": source["source_sha256"],
                "git_commit": source["git_commit"], "data": inputs, "environment": selected_runtime}
    run_file = root / "run_manifest.json"
    if run_file.is_file():
        metadata = read_json(run_file)
        if metadata.get("env_check_only"):
            if not env_only:
                raise ValueError("An env_check run cannot be resumed as a research run; choose a new run ID")
        elif env_only:
            raise ValueError("An env_check cannot resume a research run")
        if metadata.get("identity") != identity:
            raise ValueError("Requested continuation has different code, configuration, data or runtime")
    else:
        metadata = {"run_id": root.name, "purpose": "engineering" if env_only else cfg["purpose"],
                    "identity": identity, "code": source, "seed": cfg["seed"], "class_count": 10, "client_count": 4,
                    "optimizer": "SGD", "learning_rate": cfg["lr"], "epochs": None,
                    "training_unit": "fixed local steps per communication round", "stages": {},
                    "env_check_only": env_only}
        write_json(run_file, metadata)
    start = perf_counter()
    budget = cfg["budget"]
    cap = min(budget["max_seconds"], budget["max_points"]*3600/budget["points_per_hour"])
    if env_only:
        cap = min(cap, budget["env_check_seconds"])
    else:
        cap -= budget.get("reserve_seconds", 0)

    def guard():
        if perf_counter()-start >= cap:
            raise TimeoutError("Configured wall-time/point budget reached; use a compatible committed checkpoint")

    token = STEP_DEADLINE.set(guard)
    current = "env_check"

    def mark(stage, started):
        write_json(root / "progress" / f"{stage}.json", phase_receipt(root, stage, digest(identity)))
        metadata["stages"][stage] = {"status": "complete", "wall_time_sec": perf_counter()-started}
        write_json(run_file, metadata)

    def notify(arm, round_no, final):
        write_json(root / "progress" / "last_round.json", {"arm": arm, "round": round_no, "final": final})
        if budget["upload_every_rounds"] and round_no % budget["upload_every_rounds"] == 0:
            upload_checkpoint(arm)

    try:
        metadata.pop("failed_stage", None)
        write_json(root / "resolved_config.json", base_cfg)
        if not (root / "environment.json").exists():
            write_json(root / "environment.json", env)
        print(json.dumps({"mode": mode, "run": root.name, "resolved_config": str(root / "resolved_config.json"),
                         **{k: base_cfg[k] for k in ("seed", "clients", "rounds", "batch_size",
                                                      "local_steps", "kd_steps", "proxy")},
                         "gpu": env["runtime"].get("gpu")},
                         ensure_ascii=False, indent=2), flush=True)
        sources, input_seconds = timed(device, lambda: read_train_sources(base_cfg))
        started = perf_counter()
        small = {"train": max(2*base_cfg["batch_size"], 12), "val": min(64, base_cfg["limits"]["val"]),
                 "proxy": max(2*base_cfg["batch_size"], 12)}
        bench_data = setup_data({**base_cfg, "limits": small}, sources)
        metadata["split_hash"] = bench_data.split_hash
        metadata["client_config"] = base_cfg["clients"]
        metadata["rounds"] = base_cfg["rounds"]
        metadata["local_epochs_equivalent"] = base_cfg["rounds"]*base_cfg["local_steps"]*base_cfg["batch_size"]/base_cfg["limits"]["train"]
        metadata["batch_size"] = base_cfg["batch_size"]
        metadata["eval_batch_size"] = base_cfg["eval_batch_size"]
        metadata["temperature"] = base_cfg["temperature"]
        metadata["proxies"] = {"kd": base_cfg["proxy"], "frequency": base_cfg["probe2"]["domains"]}
        metadata["non_iid"] = base_cfg["split"]
        if not completed(root, "env_check", digest(identity)):
            plan = budget_plan(base_cfg, stages or MODES["precheck_all"])
            estimate = benchmark(base_cfg, bench_data, ValidationPanel(bench_data, base_cfg), device, plan)
            train_images = len(sources[0]) + (len(sources[2]) if sources[2] is not None else 0)
            estimate["official_train_images_read"] = train_images
            estimate["official_train_read_seconds"] = input_seconds
            estimate["official_train_read_images_per_second"] = train_images/input_seconds
            estimate["by_mode"] = {
                name: estimate_from_sample(base_cfg, budget_plan(base_cfg, scope), estimate)
                for name, scope in MODES.items() if scope
            }
            write_json(root / "budget_estimate.json", estimate)
            mark("env_check", started)
        pending = tuple(stage for stage in stages if not completed(root, stage, digest(identity)))
        measurement = read_json(root / "budget_estimate.json")
        base_needed = not env_only and not completed(root, "base_2x2", digest(identity))
        active = pending
        if not env_only and cfg["purpose"] == "research":
            active, estimate = affordable_prefix(base_cfg, pending, base_needed, measurement,
                                                 cap-(perf_counter()-start))
            if mode != "precheck_all" and active != pending:
                raise TimeoutError("The complete requested experiment does not fit this task; no phase training started")
        else:
            estimate = estimate_from_sample(base_cfg, budget_plan(base_cfg, pending, include_base=base_needed),
                                            measurement)
        deferred = pending[len(active):]
        estimate["selected_stages"] = active
        estimate["deferred_stages"] = deferred
        write_json(root / "invocation_budget.json", estimate)
        if not env_only and cfg["purpose"] == "research" and shutil.disk_usage(root).free < estimate["estimated_checkpoint_storage_bytes"]:
            raise RuntimeError("Output disk has less free space than the estimated checkpoint footprint")
        if env_only:
            metadata["execution_status"] = "complete"
            metadata["formal_optimizer_updates"] = 0
            metadata["checkpoint_count"] = 0
            metadata["benchmark_optimizer_updates"] = measurement["sample_local_steps"] + measurement["sample_kd_steps"]
            write_json(run_file, metadata)
            return
        guard()
        data = setup_data(base_cfg, sources)
        panel = ValidationPanel(data, base_cfg)
        current = "base_2x2"
        started = perf_counter()
        prior = completed(root, current, digest(identity))
        if prior:
            path = root / "checkpoints" / "base_2x2"
            exports, quality = path / "checkpoints", read_json(path / "quality.json")["clients"]
        else:
            path, exports, quality = base_models(base_cfg, root, data, device, panel, metadata, notify)
            mark(current, started)
            if budget["upload_each_stage"]:
                upload_checkpoint(current)

        methods = {
            "work1": lambda: run_work1(base_cfg, root, data_root, sources, data, panel, device, path, metadata, notify),
            "work2_heterogeneity": lambda: run_heterogeneity(base_cfg, root, data_root, sources, panel, device, metadata, notify),
            "work2_frequency": lambda: run_frequency(base_cfg, root, data, panel, device, exports, quality),
        }
        for current in stages:
            guard()
            if completed(root, current, digest(identity)):
                print(f"Verified and skipped {current}", flush=True)
                continue
            if current in deferred:
                break
            started = perf_counter()
            methods[current]()
            draw(root, root / "figures", current)
            mark(current, started)
            run_reports(root, metadata)
            if budget["upload_each_stage"]:
                upload_checkpoint(current)
        run_reports(root, metadata)
        if deferred:
            metadata["deferred_stages"] = list(deferred)
            raise TimeoutError(f"Budget-limited run preserved completed phases; deferred {', '.join(deferred)}")
        metadata.pop("deferred_stages", None)
        metadata["execution_status"] = "complete"
    except BaseException:
        metadata["execution_status"] = "failed"
        metadata["failed_stage"] = current
        raise
    finally:
        STEP_DEADLINE.reset(token)
        metadata["execution_wall_seconds"] = perf_counter()-start
        metadata["last_requested_mode"] = mode
        metadata["actual_updates"] = summarize_updates(root)
        write_json(run_file, metadata)


def summarize_updates(root):
    results = {}
    for path in sorted(root.rglob("rounds.csv")):
        if path.parent.name != "metrics":
            continue
        rows = read_rows(path)
        if rows:
            results[path.parent.parent.relative_to(root).as_posix()] = {
                "steps": sum(int(r["steps"]) for r in rows),
                "gpu_time_sec": sum(float(r["gpu_time_sec"]) for r in rows) if rows[0]["gpu_time_sec"] else None,
                "peak_vram_mb": max(float(r["peak_vram_mb"]) for r in rows) if rows[0]["peak_vram_mb"] else None,
            }
    for folder in sorted(root.rglob("rounds")):
        if not folder.is_dir():
            continue
        arm = folder.parent.relative_to(root).as_posix()
        if arm in results:
            continue
        tables = [p / "rounds.csv" for p in sorted(folder.iterdir()) if p.name.isdecimal()]
        rows = [row for file in tables if file.is_file() for row in read_rows(file)]
        if rows:
            results[arm] = {"steps": sum(int(r["steps"]) for r in rows),
                            "gpu_time_sec": sum(float(r["gpu_time_sec"]) for r in rows) if rows[0]["gpu_time_sec"] else None,
                            "peak_vram_mb": max(float(r["peak_vram_mb"]) for r in rows) if rows[0]["peak_vram_mb"] else None}
    for name, table in (("work1/interventions", "work1/paired_interventions.csv"),
                        ("work2_frequency", "work2_frequency/delta_r_matrix.csv")):
        if (root / table).is_file():
            rows = read_rows(root / table)
            results[name] = {"steps": sum(int(r["steps"]) for r in rows)}
            if rows and rows[0].get("gpu_time_sec"):
                results[name].update(gpu_time_sec=sum(float(r["gpu_time_sec"]) for r in rows),
                                     peak_vram_mb=max(float(r["peak_vram_mb"]) for r in rows))
    return results
