"""Export verified OpenI checkpoint runs and plan a complete follow-up task."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZIP_STORED, ZipFile

from scripts.openi_entry import file_sha, verified_files


def manifest(source):
    return json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))


def pack_recovery(source, output):
    source, output = source.resolve(), output.resolve()
    if output.suffix.lower() != ".zip" or output.is_relative_to(source):
        raise ValueError("Use a new ZIP path outside the verified run directory")
    files = verified_files(source)
    record = manifest(source)
    if record.get("purpose") != "research" or record.get("env_check_only") or not any(
        p.suffix == ".pt" for _, p in files
    ):
        raise ValueError("Recovery packaging requires a research run with committed checkpoints")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.name + ".sha256")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    with ZipFile(output, "x", compression=ZIP_STORED) as archive:
        for path, relative in files:
            archive.write(path, arcname=relative.as_posix())
        archive.write(source / "file_hashes.json", arcname="file_hashes.json")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{file_sha(output)}  {output.name}\n")
    return record["run_id"]


def unpack_recovery(source, scratch):
    """Extract only manifest-listed regular files, without trusting ZIP paths."""
    if source.is_dir():
        if (source / "file_hashes.json").is_file():
            return source
        children = [p for p in source.iterdir() if not p.is_symlink() and
                    ((p.is_dir() and (p / "file_hashes.json").is_file()) or
                     (p.is_file() and p.suffix.lower() == ".zip"))]
        if len(children) != 1:
            raise ValueError("Select one recovery ZIP or extracted run using restore_subdir")
        source = children[0]
        if source.is_dir():
            return source
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("Recovery input must be a ZIP or an extracted verified run directory")
    with ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "file_hashes.json" not in names:
            raise ValueError("Recovery ZIP has duplicate entries or no file_hashes.json")
        header = archive.getinfo("file_hashes.json")
        if header.file_size > 8_000_000:
            raise ValueError("Recovery manifest is too large")
        raw = archive.read(header)
        records = json.loads(raw)["files"]
        if set(names) != {"file_hashes.json", *records}:
            raise ValueError("Recovery ZIP differs from its file manifest")
        byte_count = sum(info.file_size for info in archive.infolist())
        if byte_count > shutil.disk_usage(scratch.parent).free:
            raise OSError("Insufficient disk space for the mounted recovery ZIP")
        scratch.mkdir()
        for name, entry in records.items():
            relative = PurePosixPath(name)
            info = archive.getinfo(name)
            if (not relative.parts or relative.is_absolute() or relative.as_posix() != name
                    or any(p.startswith(".") for p in relative.parts) or any(c in name for c in "\\:\x00")
                    or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)
                    or info.file_size != entry["bytes"]):
                raise ValueError(f"Invalid recovery ZIP member: {name!r}")
            target = scratch.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        (scratch / "file_hashes.json").write_bytes(raw)
    # restore_directory verifies the payload hashes before promoting the extracted run.
    return scratch


def a100_plan(source, *, max_seconds, max_points, points_per_hour, reserve_seconds, commit=None):
    from core.splits import validate_manifest

    files = {p.as_posix() for _, p in verified_files(source)}
    required = {"environment.json", "budget_estimate.json", "run_manifest.json", "splits_manifest.json"}
    if not required <= files:
        raise ValueError("A100 env_check evidence is incomplete")
    env = json.loads((source / "environment.json").read_text(encoding="utf-8"))
    run = manifest(source)
    budget = json.loads((source / "budget_estimate.json").read_text(encoding="utf-8"))
    split = json.loads((source / "splits_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(split, 50_000, 50_000 if split["cifar100_proxy"]["indices"] else 0, 10_000)
    if not (run.get("env_check_only") and run.get("execution_status") == "complete"
            and run.get("last_requested_mode") == "env_check" and run.get("formal_optimizer_updates") == 0
            and not run.get("checkpoint_count") and run.get("split_hash") == split.get("split_hash")):
        raise ValueError("Expected a completed, checkpoint-free env_check with matching split provenance")
    runtime = env.get("runtime", {})
    deployment = env.get("deployment", {})
    if not (env.get("cuda_kernel_verified") and "A100" in (runtime.get("gpu") or "")
            and "A100" in (env.get("nvidia_smi") or "")
            and deployment.get("managed") and deployment.get("c2net_prepare")):
        raise ValueError("No verified managed A100 evidence in this download")
    if commit and run.get("identity", {}).get("git_commit") != commit:
        raise ValueError("Downloaded run does not report the selected Git commit")
    if (len(budget["sample_forwards"]) != 4 or
            any(row["shape"] != [run["batch_size"], 10] for row in budget["sample_forwards"]) or
            any(not math.isfinite(budget[name]) or budget[name] <= 0 for name in (
                "sample_data_images_per_second", "official_train_read_images_per_second",
                "sample_local_wall_seconds", "sample_kd_wall_seconds"))):
        raise ValueError("Missing or invalid forward/throughput evidence")
    if any(not math.isfinite(value) or value <= 0 for value in (max_seconds, max_points, points_per_hour)):
        raise ValueError("Time, points and rate limits must be finite and positive")
    if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
        raise ValueError("Upload reserve must be finite and nonnegative")
    total = min(max_seconds, 3600*max_points/points_per_hour)
    usable = total - reserve_seconds
    modes = {}
    for name in ("work1_precheck", "work2_heterogeneity", "work2_frequency", "precheck_all"):
        estimate = budget["by_mode"][name]
        seconds = estimate["estimated_compute_wall_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid A100 throughput estimate for {name}")
        modes[name] = {"updates": sum(estimate["estimated_updates"].values()),
                       "compute_seconds": seconds, "with_reserve_seconds": seconds+reserve_seconds,
                       "points_with_reserve": (seconds+reserve_seconds)*points_per_hour/3600,
                       "fits": seconds <= usable,
                       "checkpoint_bytes": estimate["estimated_checkpoint_storage_bytes"]}
    # Select one whole experiment. Work 1 has priority, followed by the two Work 2 modes.
    chosen = next((name for name, row in modes.items() if name != "precheck_all" and row["fits"]), None)
    return {"measurement_run_id": run["run_id"], "reported_gpu": runtime["gpu"],
            "reported_commit": run.get("identity", {}).get("git_commit"), "task_limit_seconds": total,
            "reserved_seconds": reserve_seconds, "modes": modes, "recommended_single_mode": chosen}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack-recovery")
    pack.add_argument("--run-dir", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan-a100")
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--commit")
    plan.add_argument("--max-seconds", type=float, default=7200)
    plan.add_argument("--max-points", type=float, default=18)
    plan.add_argument("--points-per-hour", type=float, default=9)
    plan.add_argument("--reserve-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    if args.command == "pack-recovery":
        run_id = pack_recovery(args.run_dir, args.out)
        print(f"Verified recovery package for {run_id}: {args.out}")
    else:
        result = a100_plan(args.run_dir, max_seconds=args.max_seconds, max_points=args.max_points,
                           points_per_hour=args.points_per_hour, reserve_seconds=args.reserve_seconds,
                           commit=args.commit)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
