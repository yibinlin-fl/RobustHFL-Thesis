"""OpenI startup for bounded smoke, Phase 7 probes and individual Phase 8 arms."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

from scripts.openi.state import REPO


PROFILE = {"smoke": "smoke.yaml", "probe1": "probe1.yaml", "probe2": "probe2.yaml", "abcd": "abcd.yaml"}


class Tee:
    def __init__(self, console, path):
        self.console = console
        self.path = path

    def write(self, text):
        self.console.write(text)
        self.path.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.path.flush()


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=PROFILE, required=True)
    parser.add_argument("--group", choices=tuple("ABCD"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--c2net", action="store_true")
    parser.add_argument("--dataset-subdir", default=".")
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--sync-every", type=int, default=1)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration", type=Path)
    calibration.add_argument("--calibrate-batch", action="store_true")
    parser.add_argument("--git-commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-file", help="Recovery TAR.GZ relative to the c2net dataset mount")
    parser.add_argument("--require-v100", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        parser.error("Use a run ID of 1–80 letters, numbers, underscores or hyphens")
    if (args.stage == "abcd") != bool(args.group):
        parser.error("--group is required only for abcd")
    if (args.data_dir is None and not args.c2net) or args.sync_every < 1:
        parser.error("Supply --data-dir or --c2net, and a positive --sync-every")
    if not args.c2net and args.dataset_subdir != ".":
        parser.error("--dataset-subdir requires --c2net")
    if args.recovery_file and not (args.c2net and args.resume):
        parser.error("--recovery-file requires --c2net and --resume")
    if args.stage != "smoke" and args.calibration is None and not args.calibrate_batch:
        parser.error("Formal runs require --calibration from this source and GPU environment")
    if args.stage != "smoke" and not (args.c2net or args.sync_dir is not None):
        parser.error("Formal runs require --sync-dir on persistent storage or --c2net")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    return args


def prepare_context():
    from c2net.context import prepare, upload_output

    try:
        context = prepare()
    except Exception as error:
        raise RuntimeError(f"OpenI c2net prepare failed ({type(error).__name__})") from None
    if not getattr(context, "output_path", None):
        raise RuntimeError("OpenI c2net output_path unavailable")
    return context, upload_output


def restore_mounted_run(context, relative, root, run_id):
    from scripts.openi.artifacts import verify
    from scripts.openi.state import read_json

    if not getattr(context, "dataset_path", None):
        raise RuntimeError("OpenI c2net dataset_path unavailable")
    mount = Path(context.dataset_path).resolve()
    name = Path(relative)
    archive = (mount / name).resolve()
    sidecar = archive.with_name(archive.name + ".sha256")
    if (name.is_absolute() or name.suffixes[-2:] != [".tar", ".gz"]
            or not archive.is_relative_to(mount) or not sidecar.resolve().is_relative_to(mount)):
        raise ValueError("--recovery-file must be a TAR.GZ inside the c2net dataset mount")
    verify(archive, root)
    if read_json(root / "run.json")["run_id"] != run_id:
        raise ValueError("The mounted recovery run_id does not match")


def input_output(args, root, prepared=None):
    data, upload = args.data_dir, None
    outputs = []
    if args.c2net:
        context, upload = prepared or prepare_context()
        outputs.append(Path(context.output_path).resolve() / "RobustHFL")
        if data is None:
            from scripts.openi.data import locate_mounted_data

            mount = getattr(context, "dataset_path", None)
            if not mount:
                raise RuntimeError("OpenI c2net dataset_path unavailable")
            data = locate_mounted_data(Path(mount), args.dataset_subdir, root)
    if args.sync_dir:
        outputs.append(args.sync_dir.resolve())
    for output in outputs:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Sync directory and live run directory must be separate")
        if output.is_relative_to(REPO / "results"):
            raise ValueError("Sync directory must be outside existing workspace results")
        output.mkdir(parents=True, exist_ok=True)
    return data.resolve(), list(dict.fromkeys(outputs)), upload


def publish_run(root, outputs, upload, label):
    from scripts.openi.artifacts import pack

    if not outputs:
        return
    tag = f"RobustHFL_{root.name}_{label}_{uuid4().hex[:10]}.tar.gz"
    archive = pack(root, outputs[0] / tag)
    for folder in outputs[1:]:
        import shutil
        for source in (archive, archive.with_name(archive.name + ".sha256")):
            with source.open("rb") as src, (folder / source.name).open("xb") as target:
                shutil.copyfileobj(src, target)
    if upload:
        try:
            result = upload()
        except Exception as error:
            raise RuntimeError(f"OpenI c2net upload failed ({type(error).__name__})") from None
        if result is False:
            raise RuntimeError("OpenI c2net reported upload failure")
        print("c2net returned successfully; verify the downloadable archive in the task output.", flush=True)


def attempt(root, stage, task, cfg, data, device, source=None, **kwargs):
    from scripts.openi.state import write_json

    if (root / stage / "done.json").exists():
        return
    current = root / stage / "attempts" / uuid4().hex
    current.mkdir(parents=True)
    cfg = {**cfg, "out": str(current), "checkpoint_dir": str(current / "checkpoints")}
    if source is not None:
        cfg["probe2"] = {**cfg["probe2"], "sanity_image": str(current / "freq_views_sanity.png")}
        task(cfg, data, device, source)
    else:
        task(cfg, data, device, **kwargs)
    write_json(root / stage / "done.json", {"attempt": current.relative_to(root).as_posix()})


def execute(args, root, prepared=None):
    from core.data import setup_data
    from core.runtime import config, initialize
    from core.splits import digest
    from probes import probe1_local_value, probe2_freq_delta_r
    from scripts.openi.data import verify_archives
    from scripts.openi.preflight import calibrate, inspect
    from scripts.openi.state import merge_shards, read_json, source_identity, write_json
    from scripts.openi.train import run_rounds

    def status(value, **fields):
        write_json(root / "status.json", {"run_id": args.run_id, "stage": args.stage, "group": args.group,
                                          "execution_status": value, "scientific_evidence": False, **fields})

    data_root, outputs, upload = input_output(args, root, prepared)
    paths = [root, data_root, *outputs]
    report = inspect(paths, require_v100=args.require_v100)
    source = source_identity(args.git_commit)
    inputs = verify_archives(data_root)
    cfg = config(REPO / "configs" / PROFILE[args.stage], device="cuda", data_root=str(data_root), download=False,
                 manifest=str(root / "splits_manifest.json"), seed=args.seed)
    calibration = root / "calibration.json"
    if args.calibrate_batch and not calibration.exists():
        if args.resume:
            raise ValueError("Resuming requires the original calibration.json")
        result = calibrate(data_root, root, report, source)
        write_json(calibration, {**report, "calibration": result})
    calibration_input = calibration if args.calibrate_batch else args.calibration
    if calibration_input:
        measured = read_json(calibration_input)["calibration"]
        if measured["source_sha256"] != source["source_sha256"] or measured["runtime"] != report["runtime"]:
            raise ValueError("Calibration does not match the current code and CUDA environment")
        chosen = measured["trials"][-1]
        if chosen["result"] != "pass" or chosen["peak_reserved_bytes"] > report["free_vram_bytes"]*.75:
            raise ValueError("GPU no longer has the calibrated memory margin")
        for field in ("batch_size", "eval_batch_size"):
            if measured[field] != chosen[field] or not 1 <= measured[field] <= cfg[field]:
                raise ValueError(f"Invalid calibration size: {field}")
            cfg[field] = measured[field]
    cfg["out"] = str(root)
    cfg["checkpoint_dir"] = str(root / "checkpoints")
    scientific_config = {k: v for k, v in cfg.items()
                         if k not in {"data_root", "manifest", "out", "checkpoint_dir", "download"}}
    identity = {"stage": args.stage, "group": args.group, "config_sha256": digest(scientific_config),
                "source_sha256": source["source_sha256"], "git_commit": source["git_commit"],
                "runtime": report["runtime"], "data": inputs}
    if args.resume:
        if read_json(root / "run.json")["identity"] != identity:
            raise ValueError("Cannot resume with different code, data, runtime, seed or resolved config")
        if (root / "published.json").exists():
            print("Run already completed.", flush=True)
            return
    else:
        write_json(root / "run.json", {"run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                                      "identity": identity, "code": source})
        write_json(root / "resolved_config.json", cfg)
    write_json(root / f"preflight_{uuid4().hex[:10]}.json", report)
    if calibration_input and not calibration.exists():
        write_json(calibration, read_json(calibration_input))

    def checkpoint_published(round_no, final):
        status("training_committed", round=round_no)
        if final or round_no % args.sync_every == 0:
            publish_run(root, outputs, upload, f"round_{round_no:06d}_recovery")

    device = initialize(cfg)
    status("running")
    try:
        data = setup_data(cfg)
        roles = (*data.train, *data.val_clean, *data.val_mixed, *data.val_robust, *data.proxies.values())
        if any(ds.role not in {"private_train", "client_local_val", "cifar100_proxy", "in_domain_proxy"} for ds in roles):
            raise ValueError("Unexpected data role in training pipeline")
        if args.stage == "probe1":
            completed = []
            for index in range(len(data.train)):
                stage = f"probe1/client_{index}"
                if not (root / stage / "done.json").exists():
                    attempt(root, stage, probe1_local_value.run, cfg, data, device, client_indices=(index,))
                    status("client_complete", client=index)
                    publish_run(root, outputs, upload, f"client_{index}_recovery")
                completed.append(root / read_json(root / stage / "done.json")["attempt"])
            if not (root / "probe1" / "metrics").is_dir():
                merge_shards(completed, root / "probe1" / "metrics")
        else:
            name = {"smoke": "D", "probe2": "fedmd", "abcd": args.group}[args.stage]
            train_root = root / "training"
            if args.stage == "smoke" and not (train_root / "rounds" / "000001").exists():
                run_rounds(cfg, data, device, name, train_root, args.run_id, identity,
                           checkpoint_published, stop_after=1)
                initialize(cfg)
            checkpoints = run_rounds(cfg, data, device, name, train_root, args.run_id, identity,
                                    checkpoint_published)
            if args.stage in {"probe2", "smoke"}:
                status("training_complete_analysis_pending", checkpoint_dir=str(checkpoints.relative_to(root)))
                publish_run(root, outputs, upload, "pre_analysis_recovery")
                if args.stage == "smoke":
                    attempt(root, "probe1", probe1_local_value.run, cfg, data, device)
                attempt(root, "probe2", probe2_freq_delta_r.run, cfg, data, device, checkpoints)
        status("complete", acceptance="execution_only" if args.stage == "smoke" else "pending_review")
        publish_run(root, outputs, upload, "complete")
        write_json(root / "published.json", {"complete": True})
    except BaseException as error:
        status("failed", error_type=type(error).__name__)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    if args.stage == "smoke" and not args.smoke_worker:
        try:
            subprocess.run([sys.executable, "-m", "scripts.openi.launch", *argv, "--smoke-worker"],
                           cwd=REPO, timeout=170, check=True)
        except subprocess.TimeoutExpired as error:
            raise SystemExit("OpenI smoke exceeded the 170-second limit") from error
        return
    from scripts.openi.state import run_lock

    base = args.runs_root.resolve()
    root = (base / args.run_id).resolve()
    if root.parent != base:
        raise ValueError("Run ID cannot resolve outside the requested runs root")
    if root.is_relative_to(REPO / "results") or root.is_relative_to(REPO / "data"):
        raise ValueError("OpenI runs must be separate from the existing workspace results and data")
    with run_lock(base / ".locks" / args.run_id):
        prepared = None
        if args.recovery_file:
            prepared = prepare_context()
            restore_mounted_run(prepared[0], args.recovery_file, root, args.run_id)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "run.json").exists() != args.resume:
            raise ValueError("Use a new run ID or pass --resume for an existing run")
        with (root / "console.log").open("a", encoding="utf-8", buffering=1) as console:
            with redirect_stdout(Tee(sys.stdout, console)), redirect_stderr(Tee(sys.stderr, console)):
                try:
                    execute(args, root, prepared)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    raise


if __name__ == "__main__":
    main()
