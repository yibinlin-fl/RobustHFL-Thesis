"""Versioned recovery archives with a file-by-file SHA-256 manifest."""
import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from core.hashing import file_hash, stream_hash
from scripts.openi.state import commit_stage, make_stage


MANIFEST = "ARTIFACTS.json"


def members(root):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not p.is_symlink() and not any(part.startswith(".") for part in rel.parts):
            if p.name == "state.pt" and p.parent.parent.name == "rounds":
                latest = max(d.name for d in p.parent.parent.iterdir() if d.is_dir() and d.name.isdecimal())
                if p.parent.name != latest:
                    continue
            yield rel.as_posix(), p


def pack(root, output):
    root, output = root.resolve(), output.resolve()
    if not root.is_dir() or output.is_relative_to(root):
        raise ValueError("Archive output must be outside the run directory")
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(members(root))
    if not files:
        raise ValueError("Run directory is empty")
    layout = {"version": 1, "files": [
        {"path": name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for name, p in files
    ]}
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".pending-", delete=False) as pending:
        with tarfile.open(fileobj=pending, mode="w:gz", compresslevel=1) as archive:
            for name, path in files:
                archive.add(path, arcname=name, recursive=False)
            raw = json.dumps(layout, indent=2, sort_keys=True).encode() + b"\n"
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        pending.flush()
        os.fsync(pending.fileno())
    Path(pending.name).rename(output)
    checksum = file_hash(output)
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{checksum}  {output.name}\n")
    print(f"Artifact archive: {output} ({checksum})", flush=True)
    return output


def safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or path.as_posix() != name
            or any(part.startswith(".") or "\\" in part or ":" in part for part in path.parts)):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def verify(archive_path, destination=None):
    """Validate the outer SHA, reject non-files and verify each extracted byte."""
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != f"{file_hash(archive_path)}  {archive_path.name}":
        raise ValueError("Outer archive SHA-256 does not match")
    if destination is not None and destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        entries = archive.getmembers()
        names = [entry.name for entry in entries]
        if len(set(names)) != len(names) or MANIFEST not in names or not all(e.isfile() for e in entries):
            raise ValueError("Invalid archive members")
        for name in names:
            safe_name(name)
        manifest = json.load(archive.extractfile(MANIFEST))
        records = {entry["path"]: entry for entry in manifest["files"]}
        if (manifest["version"] != 1 or len(records) != len(manifest["files"])
                or set(records) != set(names) - {MANIFEST}):
            raise ValueError("Archive content differs from its manifest")
        stage = make_stage(destination.parent) if destination is not None else None
        for entry in entries:
            if entry.name == MANIFEST:
                continue
            record = records[entry.name]
            if record["bytes"] != entry.size:
                raise ValueError(f"Wrong size: {entry.name}")
            with archive.extractfile(entry) as source:
                if stage is None:
                    checksum = stream_hash(source)
                else:
                    target = stage.joinpath(*safe_name(entry.name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        shutil.copyfileobj(source, handle)
                    checksum = file_hash(target)
            if record["sha256"] != checksum:
                raise ValueError(f"SHA-256 mismatch: {entry.name}")
    if stage is not None:
        commit_stage(stage, destination)
    print(f"Verified {len(records)} files in {archive_path}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("pack")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--unpack-to", type=Path)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.run_dir, args.archive)
    else:
        verify(args.archive, args.unpack_to)


if __name__ == "__main__":
    main()
