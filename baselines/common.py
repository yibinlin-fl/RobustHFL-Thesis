"""Shared round execution and local-validation logging for heterogeneous methods."""
from pathlib import Path

from core.evaluator import evaluate_clients
from core.logger import Budget, RunLog
from core.runtime import init_client, model_state_hash, save_checkpoint
from core.splits import digest


def train_rounds(cfg, data, device, experiment, update_local, update_kd=None, log=None):
    log = log or RunLog(cfg, data, experiment)
    clients = [init_client(client["model"], cfg, device, i) for i, client in enumerate(cfg["clients"])]
    models = [model for model, _ in clients]
    if cfg["rounds"] < 1 or cfg["local_steps"] < 1:
        raise ValueError("Training requires positive round and local step counts")
    cumulative = {"steps": 0, "wall_time_sec": 0.0, "gpu_time_sec": None,
                  "peak_vram_mb": None, "comm_bytes": 0}
    hashes = None
    for round_no in range(cfg["rounds"]):
        with Budget(device) as budget:
            budget.steps = update_local(clients, round_no)
            if update_kd:
                steps, comm = update_kd(clients, round_no)
                budget.steps += steps
                budget.comm_bytes += comm
            rows, metrics = evaluate_clients(models, data, cfg, device)
        taken = budget.record()
        for key in ("steps", "wall_time_sec", "comm_bytes"):
            cumulative[key] += taken[key]
        if taken["gpu_time_sec"] is not None:
            cumulative["gpu_time_sec"] = (cumulative["gpu_time_sec"] or 0) + taken["gpu_time_sec"]
            cumulative["peak_vram_mb"] = max(cumulative["peak_vram_mb"] or 0, taken["peak_vram_mb"])
        final = round_no == cfg["rounds"] - 1
        if final:
            root = Path(cfg.get("checkpoint_dir") or "results/checkpoints") / experiment / log.run_id
            hashes = [save_checkpoint(root / f"client_{i}.pt", model, opt, cfg, data, i)
                      for i, (model, opt) in enumerate(clients)]
            log.json_event({"event": "checkpoints", "experiment": experiment,
                            "directory": str(root), "ckpt_hashes": hashes, "split_hash": data.split_hash})
        else:
            hashes = [model_state_hash(model) for model in models]
            log.json_event({"event": "round_model_states", "experiment": experiment,
                            "round": round_no + 1, "model_state_hashes": hashes, "split_hash": data.split_hash})
        for row in rows:
            log.row("clients", slice_name=experiment, round=round_no + 1, **row,
                    ckpt_hash=hashes[row["client"]])
        log.row("rounds", slice_name=experiment, round=round_no + 1, budget=taken,
                ckpt_hash=digest(hashes), **metrics)
    log.row("groups", slice_name=experiment, budget=cumulative, ckpt_hash=digest(hashes), **metrics)
    print(f"{experiment}: {cumulative['steps']} optimizer steps; checkpoints: {root}", flush=True)
    return {"metrics": metrics, "budget": cumulative, "ckpt_hash": digest(hashes), "checkpoint_dir": root}
