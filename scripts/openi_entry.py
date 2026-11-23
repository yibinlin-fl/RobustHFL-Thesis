"""Configuration-first OpenI V2 startup. Select this file in an OpenI managed task."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import mkdtemp
import traceback
from uuid import uuid4


MODES = ("env_check", "work1_precheck", "work2_heterogeneity", "work2_frequency", "precheck_all")


def repository(code_path):
    root = Path(code_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected SDK code directory: {root}")
    required = ("scripts/openi_entry.py", "configs/openi_precheck.yaml", "core/data.py")
    candidates, frontier = [], [root]
    for _ in range(4):
        for folder in frontier:
            if all((folder / path).is_file() for path in required):
                candidates.append(folder)
        if candidates:
            break
        frontier = sorted({p.resolve() for f in frontier for p in f.iterdir()
                           if p.is_dir() and not p.name.startswith(".") and p.resolve().is_relative_to(root)})
    if len(candidates) != 1:
        raise ValueError(f"Expected one RobustHFL checkout under SDK code_path: {root}; found {candidates}")
    return candidates[0]


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/openi_precheck.yaml")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--run-id", "--run_id", dest="run_id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--git-commit", "--git_commit", dest="git_commit")
    parser.add_argument("--dataset-subdir", "--dataset_subdir", dest="dataset_subdir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restore-subdir", "--restore_subdir", dest="restore_subdir",
                        help="Recovery ZIP or directory relative to the selected mount; implies --resume")
    parser.add_argument("--restore-mount", "--restore_mount", dest="restore_mount",
                        choices=("dataset", "model"), default="dataset")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--verify-output", type=Path, help="Verify downloaded hashes without launching a task")
    args = parser.parse_args(argv)
    if args.local and (args.data_dir is None or args.output_dir is None):
        parser.error("--local requires --data-dir and --output-dir")
    if not args.local and (args.data_dir or args.output_dir or args.device):
        parser.error("Managed tasks obtain their paths and CUDA device from the platform")
    if args.restore_subdir:
        args.resume = True
    return args


def published_upload():
    from c2net.context import prepare, upload_output
    return prepare(), upload_output


def plain_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_files(source):
    source = source.resolve(strict=True)
    manifest = json.loads((source / "file_hashes.json").read_text(encoding="utf-8"))
    if manifest.get("audit") == "partial":
        raise ValueError("The run has only a partial output audit")
    files = []
    for name, facts in manifest["files"].items():
        relative = Path(name)
        path = (source / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(source) or ".." in relative.parts:
            raise ValueError("Recovery manifest contains an escaping path")
        if not path.is_file() or path.stat().st_size != facts["bytes"] or file_sha(path) != facts["sha256"]:
            raise ValueError(f"Recovery file failed SHA-256: {name}")
        files.append((path, relative))
    if not files:
        raise ValueError("Output hash manifest has no files")
    return files


def restore_directory(source, dest, *, promote=False):
    files = verified_files(source)
    marker = next((p for p, relative in files if relative.as_posix() == "run_manifest.json"), None)
    if marker is None or json.loads(marker.read_text(encoding="utf-8"))["run_id"] != dest.name:
        raise ValueError("Recovery run ID does not match")
    if dest.exists():
        raise FileExistsError(dest)
    if promote:
        source.rename(dest)
        return
    staged = dest.with_name(f".restore-{uuid4().hex[:8]}")
    staged.mkdir()
    for path, relative in files:
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(source / "file_hashes.json", staged / "file_hashes.json")
    if dest.exists():
        raise FileExistsError(dest)
    staged.rename(dest)


class Tee:
    def __init__(self, console, handle):
        self.console, self.handle = console, handle

    def write(self, text):
        self.console.write(text)
        self.handle.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.handle.flush()


def report_failure(root, error):
    details = {"status": "failed", "type": type(error).__name__, "traceback_file": "execution.log"}
    plain_json(root / "failure.json", details)
    if not (root / "run_manifest.json").exists():
        plain_json(root / "run_manifest.json", {"run_id": root.name, "execution_status": "failed", **details})


def run(args, context=None, upload=None):
    from types import SimpleNamespace

    if args.verify_output:
        files = verified_files(args.verify_output)
        print(f"Verified {len(files)} downloaded files against SHA-256.", flush=True)
        return args.verify_output
    if args.local:
        context = SimpleNamespace(code_path=Path(__file__).resolve().parents[1],
                                  dataset_path=args.data_dir, output_path=args.output_dir)
        upload = None
    elif context is None:
        context, upload = published_upload()
    output = Path(context.output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A failure before configuration resolution still has a downloadable record.
    emergency = output / f"RobustHFL-startup-{uuid4().hex[:8]}"
    try:
        repo = repository(context.code_path)
        sys.path.insert(0, str(repo))
        # Read only the lightweight YAML envelope until logging is available.
        import yaml
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo / config_path
        header = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        base_config = repo / "configs/openi_precheck.yaml"
        defaults = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        mode = args.mode or header.get("mode", defaults["mode"])
        if mode not in MODES:
            raise ValueError(f"Invalid mode: {mode}")
        run_id = args.run_id or header.get("run_id", defaults["run_id"])
        if run_id == "auto":
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{mode}-{now}-{uuid4().hex[:6]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", str(run_id)):
            raise ValueError("Invalid run ID")
        if args.resume and not args.run_id and header.get("run_id", "auto") == "auto":
            raise ValueError("Resume requires the original run ID")
        candidate = output / "RobustHFL-V2" / str(run_id)
        root = candidate.resolve()
        if root != candidate or not root.is_relative_to(output):
            raise ValueError("Run directory must stay inside the platform output")
        if args.local and (output == repo / "results" or output.is_relative_to(repo / "results")):
            raise ValueError("Choose a separate local V2 output outside earlier results")
        root.parent.mkdir(parents=True, exist_ok=True)
        return run_logged(args, context, upload, repo, root, config_path, mode)
    except BaseException as error:
        if not getattr(error, "_robusthfl_reported", False):
            emergency.mkdir(exist_ok=False)
            (emergency / "execution.log").write_text(traceback.format_exc(), encoding="utf-8")
            report_failure(emergency, error)
            publish(emergency, upload, preserve_error=True)
        raise


def publish(root, upload, *, preserve_error=False):
    status_file = root / "upload_status.json"
    state = {"attempted": upload is not None, "result": "not_applicable" if upload is None else "started",
             "remote_download_verified": False}
    plain_json(status_file, state)
    if upload is None:
        return
    try:
        result = upload()
        if result is False:
            raise RuntimeError("C2NET upload_output() returned False")
        state["result"] = "returned"
        state["return_type"] = type(result).__name__
        print("C2NET upload_output() returned; verify the downloadable files and SHA-256.", flush=True)
    except BaseException as error:
        state.update(result="failed", error_type=type(error).__name__, traceback=traceback.format_exc())
        if not preserve_error:
            raise
        print("C2NET upload failed while preserving the original experiment error.", file=sys.stderr)
    finally:
        plain_json(status_file, state)


def run_logged(args, context, upload, repo, root, config_path, mode):
    from scripts.openi.state import run_lock

    with run_lock(root.parent / ".locks" / root.name):
        if args.restore_subdir:
            from scripts.openi.transfer_v2 import unpack_recovery
            from scripts.openi.v2_data import selected_folder

            location = context.dataset_path if args.restore_mount == "dataset" else getattr(
                context, "pretrain_model_path", None)
            if not location:
                raise ValueError(f"Missing C2NET {args.restore_mount} mount for recovery")
            if root.exists():
                raise FileExistsError(root)
            source = selected_folder(Path(location), args.restore_subdir)
            scratch = root.parent / f".unpack-{uuid4().hex[:8]}"
            restored = unpack_recovery(source, scratch)
            restore_directory(restored, root, promote=restored == scratch)
        if args.resume != root.exists():
            raise ValueError("Use a new run ID, or pass --resume for an existing run")
        root.mkdir(exist_ok=args.resume)
        failure = None
        with (root / "execution.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    from core.runtime import config
                    from scripts.openi.precheck_job import v2_job
                    from scripts.openi.state import source_identity
                    from scripts.openi.v2_data import discover

                    cfg = config(config_path, mode=mode, seed=args.seed, dataset_subdir=args.dataset_subdir,
                                 device=(args.device or "cuda") if args.local else "cuda")
                    if cfg["purpose"] != "research" and not args.local:
                        raise ValueError("Reduced engineering configurations are available only in --local mode")
                    if not 0 <= cfg["seed"] < 2**32:
                        raise ValueError("Seed must be in [0, 2**32)")
                    if cfg["budget"]["max_seconds"] <= 0 or cfg["budget"]["points_per_hour"] <= 0:
                        raise ValueError("Non-positive budget")
                    stage = Path(mkdtemp(prefix="robusthfl-v2-"))
                    data, inputs = discover(Path(context.dataset_path), cfg["dataset_subdir"], stage / "cifar",
                                            use_cifar100=bool(cfg["split"]["proxy100_size"]))
                    source = source_identity(args.git_commit or cfg.get("git_commit"))
                    deployment = {"managed": not args.local, "c2net_prepare": not args.local and callable(upload),
                                  "dataset_mount": str(context.dataset_path), "resolved_data_root": str(data)}
                    print(f"SDK code_path={repo}; validated input root={data}; output={root}", flush=True)
                    v2_job(cfg, root, data, inputs, source, lambda _: publish(root, upload), deployment)
                except BaseException as error:
                    failure = error
                    traceback.print_exc()
                    report_failure(root, error)
        try:
            from scripts.openi.precheck_job import audited_files
            invalid = {}
            records = audited_files(root, invalid)
            if invalid:
                plain_json(root / "audit_failure.json", {"invalid_files": invalid})
                if failure is None:
                    failure = RuntimeError("One or more output files failed validation; see audit_failure.json")
                    report_failure(root, failure)
                for name in ("audit_failure.json", "failure.json"):
                    path = root / name
                    if path.is_file():
                        records[name] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
            plain_json(root / "file_hashes.json", {"algorithm": "sha256", "files": records,
                        "audit": "valid_files_only" if invalid else "complete"})
        except BaseException as error:
            with (root / "execution.log").open("a", encoding="utf-8") as handle:
                traceback.print_exc(file=handle)
            (root / "audit_failure.json").write_text(json.dumps({"error_type": type(error).__name__}),
                                                      encoding="utf-8")
            if failure is None:
                failure = error
                report_failure(root, error)
            records = {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size}
                       for p in root.iterdir() if p.is_file() and p.name in
                       {"execution.log", "run_manifest.json", "failure.json", "audit_failure.json"}}
            plain_json(root / "file_hashes.json", {"algorithm": "sha256", "audit": "partial", "files": records})
        try:
            publish(root, upload, preserve_error=failure is not None)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            failure._robusthfl_reported = True
            raise failure
        return root


if __name__ == "__main__":
    run(arguments(sys.argv[1:]))
