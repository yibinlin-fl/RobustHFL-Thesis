"""Immutable round commits and code/runtime identity for OpenI."""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np
import torch

from core.logger import RunLog
from core.runtime import file_hash
from core.splits import digest


REPO = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("core", "baselines", "methods", "probes", "configs", "scripts")
SOURCE_FILES = ("openi_startup.py", "openi_web.py", "openi_managed.py", "requirements.txt", "requirements-openi.txt")


def source_identity(declared=None):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = git("rev-parse", "HEAD")
    git_root = git("rev-parse", "--show-toplevel")
    if not git_root or Path(git_root).resolve() != REPO:
        head = None
    if declared and (len(declared) not in (40, 64) or any(c not in "0123456789abcdef" for c in declared)):
        raise ValueError("Expected a full lowercase Git commit SHA")
    if declared and head and declared != head:
        raise ValueError("Declared Git commit differs from checked-out HEAD")
    paths = [REPO / p for p in SOURCE_FILES]
    for dirname in SOURCE_DIRS:
        paths.extend(p for p in (REPO / dirname).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".yaml", ".sh", ".ps1"})
    hashes = {p.relative_to(REPO).as_posix():
              hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(paths) if p.is_file()}
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *hashes) if head else None
    return {"git_commit": head or declared, "git_status": ("dirty" if dirty else "clean") if head
            else ("declared" if declared else "unavailable"), "source_sha256": digest(hashes),
            "source_files": hashes}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False,
                                     mode="w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)
    sync_directory(path.parent)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_stage(parent):
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def commit_stage(stage, dest):
    if dest.exists():
        raise FileExistsError(dest)
    for path in stage.rglob("*"):
        if path.is_file():
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
    sync_directory(stage)
    stage.rename(dest)
    sync_directory(dest.parent)


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as handle:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Run is active: {root}") from exc
        else:
            import msvcrt
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Run is active: {root}") from exc
        yield


def rng_state(device):
    algorithm, key, position, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": [algorithm, key.tolist(), position, has_gauss, cached],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    algorithm, key, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(key, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def write_state(path, payload):
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return file_hash(path)


def round_folders(root):
    if not root.is_dir():
        return []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdecimal())
    if [p.name for p in folders] != [f"{i:06d}" for i in range(1, len(folders)+1)]:
        raise ValueError(f"Missing or malformed committed round: {root}")
    return folders


class ShardLog(RunLog):
    def __init__(self, path, run_id, cfg, data):
        self.path = Path(path)
        self.run_id = run_id
        self.seed = cfg["seed"]
        self.hashes = {"split_hash": data.split_hash, "view_hash": data.view_hash}


def merge_shards(folders, destination):
    """Rebuild derived aggregate files from committed shards, never append on resume."""
    stage = make_stage(destination.parent)
    names = sorted({p.name for folder in folders for p in folder.iterdir()
                    if p.suffix in (".csv", ".jsonl") and p.is_file()})
    for name in names:
        paths = [folder / name for folder in folders if (folder / name).is_file()]
        with (stage / name).open("w", encoding="utf-8", newline="") as result:
            if name.endswith(".jsonl"):
                for path in paths:
                    result.write(path.read_text(encoding="utf-8"))
                continue
            writer = csv.writer(result)
            schema = None
            for path in paths:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = csv.reader(handle)
                    header = next(rows)
                    if schema is None:
                        schema = header
                        writer.writerow(schema)
                    if schema != header:
                        raise ValueError(f"CSV schema changed: {path}")
                    writer.writerows(rows)
    commit_stage(stage, destination)
