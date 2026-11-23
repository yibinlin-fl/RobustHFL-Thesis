"""Exploratory V2 experiments. All validation endpoints are training-sourced holdouts."""
import csv
from copy import deepcopy
from itertools import product
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from core.data import setup_data
from core.evaluator import local_accuracy, ranking
from core.logger import Budget
from core.runtime import batch_at, file_hash, init_client, normalize, read_checkpoint, step
from core.splits import digest
from methods.work1_local_value.local import LocalHistory, observe
from probes import probe1_local_value, probe2_freq_delta_r
from scripts.openi.precheck_protocol import (
    BLUR, GAUSSIAN, SCENARIOS, STRATEGIES, qualify, state_file, train_arm, training_config,
)
from scripts.openi.state import read_json, write_json


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"No records generated for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    with pending.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
    pending.replace(path)


def endpoint(model, i, data, cfg, device, panel):
    return (local_accuracy(model, data.val_clean[i], cfg, device),
            panel.robust(model, i, cfg, device))


def load_export(root, data, cfg, device):
    return [read_checkpoint(root / f"client_{i}.pt", cfg, device, i, data.split_hash, data.view_hash)
            for i in range(len(data.train))]


def model_gate(root, data, cfg, device, checkpoints):
    models = [m for m, _ in load_export(checkpoints, data, cfg, device)]
    checks = qualify(models, data, cfg, device)
    write_json(root / "quality.json", {"purpose": cfg["purpose"], "clients": checks})
    if cfg["purpose"] == "research" and not all(row["passed"] for row in checks):
        raise RuntimeError(f"Classification quality gate failed. See {root / 'quality.json'}")
    return checks


def base_models(cfg, root, data, device, panel, metadata, notify):
    path = root / "checkpoints" / "base_2x2"
    warm = cfg["work1"]["warmup_round"]
    if not 1 <= warm < cfg["rounds"]:
        raise ValueError("Work 1 needs separate warm-up and analysis checkpoints")
    retain = {warm, warm-1, cfg["rounds"]-1}
    exports = train_arm(cfg, data, panel, device, path, "base_2x2", metadata, notify, retain=retain)
    checks = model_gate(path, data, cfg, device, exports)
    snapshots = {label: str(state_file(path, round_no).relative_to(root))
                 for label, round_no in (("warmup", warm), ("analysis", cfg["rounds"]))}
    curves = read_rows(path / "metrics" / "rounds.csv")
    write_json(path / "snapshots.json", {"paths": snapshots, "quality": checks,
                "first_ce": float(curves[0]["local_ce_loss"]), "last_ce": float(curves[-1]["local_ce_loss"])})
    return path, exports, checks


def load_snapshot(base, round_no, i, cfg, device):
    model, opt = init_client(cfg["clients"][i]["model"], cfg, device, i)
    if round_no:
        record = torch.load(state_file(base, round_no), map_location="cpu", weights_only=True)
        payload = record["clients"][i]
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
    return model, opt


def sample_signals(scores, ds, i, phase, architecture, corruption):
    for pos, idx in enumerate(scores.ids.tolist()):
        spec = ds.assignments.get(idx)
        yield {"phase": phase, "client": i, "architecture": architecture, "client_corruption": corruption,
               "train_index": idx, "actual_corruption": spec.kind if spec else "clean",
               "severity": spec.severity if spec else 0,
               **{name: float(values[pos]) if np.isfinite(values[pos]) else None
                  for name, values in scores.raw.items()}}


def paired_effects(model, opt, scores, ds, i, phase, data, cfg, device, panel, checkpoint):
    output, correlations = [], []
    options = cfg["work1"]
    for signal, values in scores.raw.items():
        signal_rows = []
        for repeat in range(options["repeats"]):
            seed = cfg["seed"] + repeat + i*300
            for quantile, selected, control, matched in probe1_local_value.paired_groups(
                    ds, values, scores.ids, cfg, seed):
                rng = np.random.default_rng(seed+quantile)
                eligible = sorted(set(ds.ids)-set(selected)-set(control))
                background = rng.choice(eligible, options["background_size"], replace=False).tolist()
                ids = selected + control + background
                shuffled = rng.permutation(len(ids))
                x, y, _ = batch_at(ds, [ds.position(ids[j]) for j in shuffled], device)
                n = len(selected)
                scores_mean = float(np.mean(values[[ds.position(idx) for idx in selected]]))
                outcomes = {}
                usage = {}
                for arm, target, weight in (
                    ("normal", np.zeros(len(ids), dtype=bool), 1.0), ("downweight", shuffled < n, cfg["downweight"]["weight"]),
                    ("zero", shuffled < n, 0.0), ("random_downweight", (shuffled >= n)&(shuffled < 2*n),
                        cfg["downweight"]["weight"]), ("random_zero", (shuffled >= n)&(shuffled < 2*n), 0.0),
                ):
                    with Budget(device) as budget:
                        branch, branch_opt = deepcopy((model, opt))
                        branch.eval()
                        weights = torch.ones(len(ids), device=device)
                        weights[torch.as_tensor(target, device=device)] = weight
                        for _ in range(options["steps"]):
                            ce = F.cross_entropy(branch(normalize(x)), y, reduction="none")
                            step((weights*ce).mean(), branch, branch_opt, cfg["clip"])
                        outcomes[arm] = endpoint(branch, i, data, cfg, device, panel)
                        budget.steps = options["steps"]
                    usage[arm] = budget.record()
                for arm, (clean, robust) in outcomes.items():
                    reference = {"downweight": "random_downweight", "zero": "random_zero"}.get(arm)
                    row = {"phase": phase, "client": i, "architecture": cfg["clients"][i]["model"],
                           "corruption": cfg["clients"][i]["corruption"]["kind"], "signal": signal, "quantile": quantile,
                           "repeat": repeat, "arm": arm, **usage[arm], "lr": cfg["lr"],
                           "checkpoint_sha256": checkpoint, "ordered_batch_sha256": digest([ids[j] for j in shuffled.tolist()]),
                           "group_size": n, "batch_size": len(ids), "control_label_match": matched,
                           "score": scores_mean, "clean_acc": clean, "robust_acc": robust,
                           "gain_clean": clean-outcomes["normal"][0], "gain_robust": robust-outcomes["normal"][1],
                           "gain_vs_matched_control": robust-outcomes[reference][1] if reference else None}
                    output.append(row)
                    signal_rows.append(row)
        for arm in ("downweight", "zero", "random_downweight", "random_zero"):
            rows = [r for r in signal_rows if r["arm"] == arm]
            stats = ranking([r["score"] for r in rows], [r["gain_robust"] for r in rows])
            correlations.append({"phase": phase, "client": i, "architecture": cfg["clients"][i]["model"],
                                 "signal": signal, "arm": arm, **stats})
    return output, correlations


def run_work1(cfg, root, input_root, sources, data, panel, device, base, metadata, notify):
    dest = root / "work1"
    curves, summaries = [], []
    for strategy in STRATEGIES:
        arm_cfg = training_config(cfg, root, input_root, strategy)
        arm_data = setup_data(arm_cfg, sources) if strategy == "clean_ce" else data
        arm = base if strategy == "mixed_ce" else dest / "training" / strategy
        checks = (read_json(arm / "quality.json")["clients"] if strategy == "mixed_ce" else
                  model_gate(arm, arm_data, arm_cfg, device,
                             train_arm(arm_cfg, arm_data, panel, device, arm, strategy, metadata, notify,
                                       strategy=strategy)))
        rows = read_rows(arm / "metrics" / "rounds.csv")
        recorded = [int(row["steps"]) for row in rows]
        if recorded != [cfg["local_steps"]*len(data.train)]*cfg["rounds"]:
            raise ValueError(f"Work 1 budget mismatch for {strategy}")
        curves.extend({"strategy": strategy, **r} for r in rows)
        summaries.append({"strategy": strategy, "steps": sum(recorded),
                          "downweighted_visits": sum(int(r["downweighted_visits"]) for r in rows),
                          **{f: rows[-1][f] for f in ("clean_acc", "corrupt_acc", "worst_client_acc")},
                          "quality_clients_passed": sum(c["passed"] for c in checks)})
    if summaries[2]["downweighted_visits"] != summaries[3]["downweighted_visits"]:
        raise ValueError("High-loss/random downweight exposure budgets differ")
    for row in summaries:
        row["robust_vs_mixed_ce"] = float(row["corrupt_acc"])-float(summaries[1]["corrupt_acc"])
    write_rows(dest / "training_curves.csv", curves)
    write_rows(dest / "clean_mixed_comparison.csv", summaries[:2])
    write_rows(dest / "loss_random_comparison.csv", summaries[1:])
    signals, interventions, associations, detection, skipped = [], [], [], [], []
    for phase, round_no in (("warmup", cfg["work1"]["warmup_round"]), ("analysis", cfg["rounds"])):
        origins = [load_snapshot(base, round_no, i, cfg, device) for i in range(len(data.train))]
        checks = qualify([m for m, _ in origins], data, cfg, device)
        checksum = file_hash(state_file(base, round_no))
        for i, (model, opt) in enumerate(origins):
            if cfg["purpose"] == "research" and not checks[i]["passed"]:
                skipped.append({"phase": phase, "client": i, **checks[i]})
                continue
            ds = data.train[i]
            history = LocalHistory()
            previous, _ = load_snapshot(base, round_no-1, i, cfg, device)
            observe(previous, ds, history, cfg, device, round_no-1, cfg["seed"]+i, score=False)
            del previous
            scores = observe(model, ds, history, cfg, device, round_no, cfg["seed"]+i)
            rows = list(sample_signals(scores, ds, i, phase, cfg["clients"][i]["model"], data.corruptions[i].kind))
            signals.extend(rows)
            severe = [r["severity"] == cfg["mixture"]["severe"] for r in rows]
            auc = float(roc_auc_score(severe, scores.raw["loss_i"])) if 0 < sum(severe) < len(severe) else None
            detection.append({"phase": phase, "client": i, "architecture": cfg["clients"][i]["model"],
                              "loss_severe_detection_auroc": auc, "n_severe": sum(severe), "n": len(severe)})
            results, stats = paired_effects(model, opt, scores, ds, i, phase, data, cfg, device, panel, checksum)
            interventions.extend(results)
            associations.extend(stats)
    write_json(dest / "skipped_snapshots.json", skipped)
    write_rows(dest / "sample_signals.csv", signals)
    write_rows(dest / "paired_interventions.csv", interventions)
    write_rows(dest / "signal_value_correlations.csv", associations)
    write_rows(dest / "corruption_detection.csv", detection)


def run_heterogeneity(cfg, root, input_root, sources, panel, device, metadata, notify):
    dest = root / "work2_heterogeneity"
    summaries, individual, gains = [], [], []
    for scenario in SCENARIOS:
        arm_cfg = training_config(cfg, root, input_root, scenario)
        data = setup_data(arm_cfg, sources)
        comparison = {}
        for method in ("local_ce", "fedmd"):
            arm = dest / "experiment_logs" / scenario / method
            checkpoint = train_arm(arm_cfg, data, panel, device, arm, f"{scenario}/{method}", metadata, notify,
                                   method=method)
            model_gate(arm, data, arm_cfg, device, checkpoint)
            summary, = read_rows(arm / "metrics" / "groups.csv")
            expected = len(data.train)*cfg["rounds"]*(cfg["local_steps"]+(cfg["kd_steps"] if method == "fedmd" else 0))
            if int(summary["steps"]) != expected:
                raise ValueError("Heterogeneity budget mismatch")
            counts = [{"client": i, "clean": len(ds)-len(ds.assignments),
                       **{k: sum(s.kind == k for s in ds.assignments.values()) for k in (GAUSSIAN, BLUR)}}
                      for i, ds in enumerate(data.train)]
            write_json(arm.parent / "training_composition.json", {"counts": counts,
                       "clients": arm_cfg["clients"], "split_hash": data.split_hash, "view_hash": data.view_hash})
            summaries.append({"scenario": scenario, "method": method, **summary})
            rows = [r for r in read_rows(arm / "metrics" / "clients.csv") if int(r["round"]) == cfg["rounds"]]
            individual.extend({"scenario": scenario, "method": method, **r} for r in rows)
            comparison[method] = {r["client"]: r for r in rows}
        for i, local in comparison["local_ce"].items():
            kd = comparison["fedmd"][i]
            gains.append({"scenario": scenario, "client": i, "architecture": local["model"],
                          "local_clean": local["clean_acc"], "local_robust": local["corrupt_acc"],
                          "kd_clean": kd["clean_acc"], "kd_robust": kd["corrupt_acc"],
                          "kd_gain": float(kd["corrupt_acc"])-float(local["corrupt_acc"])})
    write_rows(dest / "heterogeneity_main_results.csv", summaries)
    write_rows(dest / "kd_collaboration_gains.csv", gains)
    write_rows(dest / "client_level_results.csv", individual)


def run_frequency(cfg, root, data, panel, device, exports, quality):
    dest = root / "work2_frequency"
    attempt = dest / "probe"
    if not (attempt / "complete.json").exists():
        # An incomplete attempt is preserved outside the canonical path.
        if attempt.exists():
            from uuid import uuid4
            attempt.rename(attempt.with_name(f"incomplete-{uuid4().hex[:8]}"))
        frequency_cfg = {**cfg, "out": str(attempt), "probe2": {**cfg["probe2"], "save_view_pixels": True,
                         "sanity_image": str(dest / "frequency_views.png")}}
        probe2_freq_delta_r.run(frequency_cfg, data, device, exports,
                                 robust_eval=lambda m, i: panel.robust(m, i, cfg, device))
        write_json(attempt / "complete.json", {"split_hash": data.split_hash, "view_hash": data.view_hash})
    shutil.copyfile(attempt / "student_teacher_band_gain.csv", dest / "delta_r_matrix.csv")
    matrix = read_rows(dest / "delta_r_matrix.csv")
    count = len(data.train)
    expected = count*(count-1)*3*cfg["probe2"]["repeats"]*len(cfg["probe2"]["domains"])
    if len(matrix) != expected or {int(row["steps"]) for row in matrix} != {cfg["probe2"]["delta_steps"]}:
        raise ValueError("Incomplete or mismatched frequency interventions")
    per_batch = {(r["slice"], r["replicate"]): r["public_batch_sha256"] for r in matrix}
    if any(row["public_batch_sha256"] != per_batch[(row["slice"], row["replicate"])] for row in matrix):
        raise ValueError("Frequency pairing used different public batches")
    metadata = [{"client": i, "architecture": cfg["clients"][i]["model"],
                 "training_corruption": data.corruptions[i].kind, "training_rounds": cfg["rounds"],
                 "checkpoint_sha256": file_hash(exports / f"client_{i}.pt"), **check}
                for i, check in enumerate(quality)]
    write_rows(dest / "teacher_student_metadata.csv", metadata)
    candidates = ("mi", "confidence", "teacher_marginal_entropy", "cross_view_consistency",
                  "teacher_consensus", "discrepancy")
    keys = ("slice", "student", "teacher", "band", "replicate", "teacher_argmax_classes",
            "teacher_majority_fraction", *candidates)
    write_rows(dest / "unlabeled_reliability_scores.csv", ({key: r[key] for key in keys} for r in matrix))
    ranks = []
    for domain, student, band, signal in product(cfg["probe2"]["domains"], ("all", *map(str, range(count))),
                                               ("all", "full", "low", "high"), candidates):
        rows = [r for r in matrix if r["slice"] == domain and student in ("all", r["student"])
                and band in ("all", r["band"]) and r[signal]]
        metrics = ranking([float(r[signal]) for r in rows], [float(r["delta_r"]) for r in rows])
        ranks.append({"domain": domain, "student": student, "band": band, "signal": signal, **metrics})
    write_rows(dest / "score_delta_r_correlations.csv", ranks)
    write_json(dest / "runtime_statistics.json", {"arms": len(matrix),
               "steps": sum(int(r["steps"]) for r in matrix),
               "gpu_time_sec": sum(float(r["gpu_time_sec"]) for r in matrix) if matrix[0]["gpu_time_sec"] else None,
               "peak_vram_mb": max(float(r["peak_vram_mb"]) for r in matrix) if matrix[0]["peak_vram_mb"] else None})
