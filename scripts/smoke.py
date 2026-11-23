"""Run all smoke stages in one process under an enforced timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

def worker(data_root, root, adapters, requested_device):
    started = perf_counter()
    import pytest
    from baselines import aughfl_adapter, fedmd_kd, local_ce, rahfl_adapter
    from core.data import setup_data
    from core.runtime import config, initialize
    from methods import joint_framework
    from probes import probe1_local_value, probe2_freq_delta_r
    if pytest.main(["-q", "tests"]) != 0:
        return 1
    cfg = config("configs/smoke.yaml", data_root=data_root, device=requested_device)
    device = initialize(cfg)
    if device.type == "cuda":
        cfg["probe2"]["sanity_image"] = str(root / "freq_views_sanity.png")
    print(f"Smoke device: {device}", flush=True)
    data = setup_data(cfg)
    print(f"Smoke split SHA-256: {data.split_hash}", flush=True)
    stage = lambda name: {**cfg, "out": str(root / name)}
    result = fedmd_kd.run(stage("fedmd"), data, device)
    probe1_local_value.run(stage("probe1"), data, device)
    probe2_freq_delta_r.run(stage("probe2"), data, device, result["checkpoint_dir"])
    abcd = config("configs/abcd_smoke.yaml", data_root=data_root, out=str(root / "abcd"), device=requested_device)
    joint_framework.run(abcd, data, device, abcd["groups"])
    if adapters:
        for module in (local_ce, rahfl_adapter, aughfl_adapter):
            module.run(stage(module.__name__.rsplit(".", 1)[-1]), data, device)
    result = {"duration_sec": perf_counter()-started, "split_hash": data.split_hash,
              "view_hash": data.view_hash, "checkpoints": str(result["checkpoint_dir"]),
              "checkpoint_set_sha256": result["ckpt_hash"], "adapters": adapters, "device": str(device)}
    with (root / "smoke_runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/runs/smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--with-adapters", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.data_root, Path(args.out), args.with_adapters, args.device)
    try:
        return subprocess.run([sys.executable, "-m", "scripts.smoke", *sys.argv[1:], "--worker"],
                              timeout=170, check=False).returncode
    except subprocess.TimeoutExpired:
        print("Smoke stopped at the 170-second watchdog limit.", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
