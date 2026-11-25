"""Read the allocated GPU and filesystem, and calibrate with a bounded CIFAR workload."""
import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch

from core.runtime import config, init_client, initialize
from scripts.openi.state import REPO, make_stage, source_identity, write_json


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def runtime_identity():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    drivers = command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    packages = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "Pillow", "PyYAML", "scikit-learn", "c2net"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "os": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "packages": packages,
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "driver_versions": sorted(set(drivers.splitlines())) if drivers else [],
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "gpu_bytes": props.total_memory,
    }


def inspect(paths, *, require_v100=False):
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}, CUDA build={torch.version.cuda}")
    runtime = runtime_identity()
    if require_v100 and ("V100" not in runtime["gpu"] or runtime["compute_capability"] != [7, 0]):
        raise RuntimeError(f"Selected GPU is not a V100: {runtime['gpu']}")
    import torchvision
    from torchvision.ops import nms

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8).requires_grad_()
    (x @ x.T).mean().backward()
    torch.fft.rfft2(x.detach())
    nms(torch.tensor([[0., 0., 1., 1.]], device=device), torch.ones(1, device=device), .5)
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    disks = {}
    for path in paths:
        path = Path(path).resolve()
        usage = shutil.disk_usage(path)
        disks[str(path)] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    return {
        "runtime": runtime, "cuda_kernel_verified": True,
        "cuda_visible_count": torch.cuda.device_count(), "logical_device": device.index,
        "free_vram_bytes": free, "cuda_total_bytes": total, "architectures": torch.cuda.get_arch_list(),
        "python_executable": sys.executable, "torchvision_import": torchvision.__version__,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader"), "disk": disks,
    }


def calibrate(data_root, storage, report, source):
    from baselines.local_ce import local_seed
    from core.data import setup_data
    from core.evaluator import evaluate_clients
    from methods.work1_local_value.local import train_local
    from methods.work2_freq_kd.kd import frequency_update

    cfg = config(REPO / "configs/base.yaml", device="cuda", data_root=str(data_root), download=False)
    max_batch, max_eval = cfg["batch_size"], cfg["eval_batch_size"]
    cfg.update(local_steps=1, kd_steps=1, limits={"train": max_batch, "val": max_eval, "proxy": max_batch})
    trials = []
    device = initialize(cfg)
    calibration_files = make_stage(storage)
    cfg["manifest"] = str(calibration_files / "splits_manifest.json")
    data = setup_data(cfg)
    for divisor in (1, 2, 4, 8):
        batch, evaluation = max_batch // divisor, max_eval // divisor
        cfg.update(batch_size=batch, eval_batch_size=evaluation)
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats(device)
        clients, comparison, model, opt = None, None, None, None
        try:
            clients = [init_client(client["model"], cfg, device, i) for i, client in enumerate(cfg["clients"])]
            for i, (model, opt) in enumerate(clients):
                train_local(model, opt, data.train[i], cfg, device, local_seed(cfg, i, 0), 1,
                            "Fixed Consistency")
            comparison = deepcopy(clients)
            frequency_update(cfg, data, device)(comparison, 0)
            evaluate_clients([m for m, _ in comparison], data, cfg, device)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
            result = "pass" if peak <= free * .75 else "insufficient_margin"
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_reserved(device)
            result = "out_of_memory"
        finally:
            del clients, comparison, model, opt
            torch.cuda.empty_cache()
        trials.append({"batch_size": batch, "eval_batch_size": evaluation, "free_bytes_before": free,
                       "peak_reserved_bytes": peak, "result": result})
        if result == "pass":
            return {
                "source_sha256": source["source_sha256"], "runtime": report["runtime"],
                "basis": "four CIFAR clients, one local consistency step and one multi-view KD step with duplicate clients",
                "batch_size": batch, "eval_batch_size": evaluation, "trials": trials,
            }
    raise RuntimeError(f"No calibrated batch met the memory margin: {trials}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--require-v100", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out already exists")
    if args.calibrate and args.data_dir is None:
        parser.error("--calibrate requires official CIFAR archives via --data-dir")
    args.runs_root.mkdir(parents=True, exist_ok=True)
    report = inspect([args.runs_root, *([args.data_dir] if args.data_dir else [])],
                     require_v100=args.require_v100)
    report["code"] = source_identity()
    if args.calibrate:
        report["calibration"] = calibrate(args.data_dir, args.runs_root, report, report["code"])
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "code"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
