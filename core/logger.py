"""Append-only run tables with a matched JSONL record for each CSV row."""
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from core.splits import digest


COMMON = ("run_id", "timestamp_utc", "seed", "slice", "steps", "wall_time_sec", "gpu_time_sec",
          "peak_vram_mb", "comm_bytes", "split_hash", "view_hash", "ckpt_hash")


class Budget:
    def __init__(self, device):
        self.device = device
        self.steps = 0
        self.comm_bytes = 0

    def __enter__(self):
        self.start = perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_start = torch.cuda.Event(enable_timing=True)
            self.gpu_end = torch.cuda.Event(enable_timing=True)
            self.gpu_start.record(torch.cuda.current_stream(self.device))
        return self

    def __exit__(self, *exc):
        self.gpu_time = None
        self.peak_vram = None
        if self.device.type == "cuda":
            self.gpu_end.record(torch.cuda.current_stream(self.device))
            self.gpu_end.synchronize()
            self.gpu_time = self.gpu_start.elapsed_time(self.gpu_end) / 1000
            self.peak_vram = torch.cuda.max_memory_allocated(self.device) / 1024**2
        self.wall_time = perf_counter() - self.start

    def record(self):
        return {"steps": self.steps, "comm_bytes": self.comm_bytes, "wall_time_sec": self.wall_time,
                "gpu_time_sec": self.gpu_time, "peak_vram_mb": self.peak_vram}


def tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class RunLog:
    def __init__(self, cfg, data, experiment):
        self.path = Path(cfg["out"])
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid4().hex
        self.seed = cfg["seed"]
        self.hashes = {"split_hash": data.split_hash, "view_hash": data.view_hash}
        self.json_event({"event": "config", "experiment": experiment, "config": cfg,
                         "config_hash": digest(cfg), "torch_cuda_build": torch.version.cuda,
                         "versions": {name: version(name) for name in
                                      ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "PyYAML", "Pillow")},
                         **self.hashes})

    def json_event(self, event):
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": self.run_id, **event}, ensure_ascii=False, allow_nan=False) + "\n")

    def row(self, table, *, slice_name, budget=None, ckpt_hash=None, **values):
        row = {key: None for key in COMMON}
        row.update(run_id=self.run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(), seed=self.seed,
                   slice=slice_name, ckpt_hash=ckpt_hash, **self.hashes)
        if budget is not None:
            row.update(budget)
        row.update(values)
        path = self.path / f"{table}.csv"
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle)) != list(row):
                    raise ValueError(f"Schema mismatch for {path}")
            header = False
        else:
            header = True
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if header:
                writer.writeheader()
            writer.writerow(row)
        self.json_event({"event": table, **row})
