"""Round-resumable federated training using the existing experiment primitives."""
from pathlib import Path

import torch

from baselines.fedmd_kd import uniform_update
from baselines.local_ce import ce_update, local_seed
from core.evaluator import evaluate_clients
from core.logger import Budget
from core.runtime import file_hash, init_client, model_state_hash, save_checkpoint
from core.splits import digest
from methods.work1_local_value.local import LocalHistory, observe, train_local
from methods.work2_freq_kd.kd import frequency_update
from scripts.openi.state import (
    ShardLog, commit_stage, make_stage, merge_shards, read_json, restore_rng,
    rng_state, round_folders, write_json, write_state,
)


def update_value(clients, histories, cfg, data, device, round_no):
    steps = 0
    for i, (model, opt) in enumerate(clients):
        seed = local_seed(cfg, i, round_no)
        enabled = round_no >= cfg["local"]["warmup_rounds"]
        scores = observe(model, data.train[i], histories[i], cfg, device, round_no, seed, score=enabled)
        mode = "Candidate Quality-Modulated" if enabled else "CE"
        steps += train_local(model, opt, data.train[i], cfg, device, seed, cfg["local_steps"],
                             mode, scores, cfg["local"]["scale"])
    return steps


def run_rounds(cfg, data, device, name, root, run_id, identity, publish, *, stop_after=None,
               local_update=None, evaluator=evaluate_clients, retain_rounds=(), compact=False):
    if name not in {"A", "B", "C", "D", "fedmd", "local_ce"}:
        raise ValueError(f"Invalid training arm: {name}")
    modulated = name in {"B", "D"}
    if min(cfg["rounds"], cfg["local_steps"], cfg["kd_steps"]) < 1:
        raise ValueError("Training budgets must be positive")
    if modulated and not 1 <= cfg["local"]["warmup_rounds"] < cfg["rounds"]:
        raise ValueError("Expected warm-up and local-value rounds")
    if stop_after is not None and not 1 <= stop_after <= cfg["rounds"]:
        raise ValueError("Requested stop round is outside the configured budget")

    rounds = root / "rounds"
    committed = round_folders(rounds)
    trained = len(committed)
    identity = {**identity, "training_arm": name, "split_hash": data.split_hash, "view_hash": data.view_hash}
    identity_hash = digest(identity)
    cumulative = {"steps": 0, "comm_bytes": 0, "wall_time_sec": 0.0,
                  "gpu_time_sec": None, "peak_vram_mb": None}
    clients = [init_client(c["model"], cfg, device, i) for i, c in enumerate(cfg["clients"])]
    histories = [LocalHistory() for _ in clients] if modulated else []
    if trained:
        receipt = read_json(committed[-1] / "receipt.json")
        checkpoint = committed[-1] / "state.pt"
        if receipt["identity_sha256"] != identity_hash or receipt["state_sha256"] != file_hash(checkpoint):
            raise ValueError("Round checkpoint does not match code, config, runtime, data or saved SHA-256")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["completed_rounds"] != trained or state["identity_sha256"] != identity_hash:
            raise ValueError("Round state does not match its receipt")
        for (model, opt), saved in zip(clients, state["clients"], strict=True):
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
        for history, saved in zip(histories, state["local_histories"], strict=True):
            history.prototypes = {i: value.to(device) for i, value in saved["prototypes"].items()}
            history.losses = saved["losses"]
        restore_rng(state["rng"], device)
        cumulative = state["cumulative"]
        print(f"Resumed {name} after committed round {trained}", flush=True)

    update_local = local_update or ce_update(cfg, data, device)
    update_public = (frequency_update if name in {"C", "D"} else uniform_update)(cfg, data, device)
    for round_no in range(trained, stop_after or cfg["rounds"]):
        with Budget(device) as budget:
            local = (update_value(clients, histories, cfg, data, device, round_no) if modulated
                     else update_local(clients, round_no))
            local_steps, extra = local if isinstance(local, tuple) else (local, {})
            budget.steps = local_steps
            steps, comm = update_public(clients, round_no) if name != "local_ce" else (0, 0)
            budget.steps += steps
            budget.comm_bytes += comm
            rows, metrics = evaluator([model for model, _ in clients], data, cfg, device)
        totals = budget.record()
        for key in ("steps", "comm_bytes", "wall_time_sec"):
            cumulative[key] += totals[key]
        if device.type == "cuda":
            cumulative["gpu_time_sec"] = (cumulative["gpu_time_sec"] or 0) + totals["gpu_time_sec"]
            cumulative["peak_vram_mb"] = max(cumulative["peak_vram_mb"] or 0, totals["peak_vram_mb"])

        stage = make_stage(rounds)
        final = round_no + 1 == cfg["rounds"]
        hashes = [model_state_hash(model) for model, _ in clients]
        log = ShardLog(stage, run_id, cfg, data)
        for row in rows:
            log.row("clients", slice_name=name, round=round_no+1, **row, ckpt_hash=hashes[row["client"]])
        log.row("rounds", slice_name=name, round=round_no+1, budget=totals,
                ckpt_hash=digest(hashes), **metrics, **extra)
        if final:
            log.row("groups", slice_name=name, budget=cumulative, ckpt_hash=digest(hashes), **metrics)
        state = {
            "completed_rounds": round_no+1, "identity_sha256": identity_hash,
            "clients": [{"model": model.state_dict(), "optimizer": opt.state_dict()} for model, opt in clients],
            "local_histories": [vars(history) for history in histories],
            "rng": rng_state(device), "cumulative": cumulative,
        }
        checksum = write_state(stage / "state.pt", state)
        write_json(stage / "receipt.json", {"completed_rounds": round_no+1, "identity_sha256": identity_hash,
                                            "state_sha256": checksum, "model_sha256": hashes})
        commit_stage(stage, rounds / f"{round_no+1:06d}")
        if compact and round_no >= 1 and round_no not in retain_rounds:
            previous = rounds / f"{round_no:06d}" / "state.pt"
            previous.unlink(missing_ok=True)
        publish(round_no+1, final)
        print(f"{name}: committed round {round_no+1}/{cfg['rounds']} ({checksum})", flush=True)

    if len(round_folders(rounds)) != cfg["rounds"]:
        return None
    export = root / "checkpoints"
    if not export.is_dir():
        stage = make_stage(root)
        hashes = [save_checkpoint(stage / f"client_{i}.pt", model, opt, cfg, data, i)
                  for i, (model, opt) in enumerate(clients)]
        write_json(stage / "manifest.json", {"client_sha256": hashes, "split_hash": data.split_hash,
                                             "view_hash": data.view_hash, "identity_sha256": identity_hash})
        commit_stage(stage, export)
    manifest = read_json(export / "manifest.json")
    if manifest["identity_sha256"] != identity_hash or any(
        file_hash(export / f"client_{i}.pt") != checksum
        for i, checksum in enumerate(manifest["client_sha256"])
    ) or len(manifest["client_sha256"]) != len(clients):
        raise ValueError("Exported client checkpoints differ from their manifest")
    if not (root / "metrics").is_dir():
        merge_shards(round_folders(rounds), root / "metrics")
    return export
