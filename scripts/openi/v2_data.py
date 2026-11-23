"""Resolve verified CIFAR inputs inside an OpenI mount without extracting arbitrary members."""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from zipfile import ZIP_STORED, ZipFile

from torchvision.datasets import CIFAR10, CIFAR100

from core.data import md5_file
from core.runtime import config, file_hash
from scripts.openi.data import SIZES, verify_archives
from scripts.openi.state import commit_stage, make_stage, read_json, write_json


def required_sources(use_cifar100):
    return (CIFAR10, CIFAR100) if use_cifar100 else (CIFAR10,)


def official_members(classes):
    return {f"{cls.base_folder}/{name}": md5 for cls in classes for name, md5 in cls.train_list}


def validate_tree(root, classes):
    records = {}
    for cls in classes:
        archive = root / cls.filename
        if archive.is_file():
            sources = [(archive, cls.tgz_md5)]
        else:
            sources = [(root / cls.base_folder / name, md5) for name, md5 in cls.train_list]
        for path, md5 in sources:
            if not path.is_file() or md5_file(path) != md5:
                raise ValueError(f"Invalid or missing official CIFAR input: {path}")
            records[path.relative_to(root).as_posix()] = {
                "md5": md5, "sha256": file_hash(path), "bytes": path.stat().st_size,
            }
    return records


def selected_folder(mount, relative):
    mount = mount.resolve(strict=True)
    path = (mount / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(mount):
        raise ValueError("Input path must be relative to and inside the data mount")
    return path


def safe_parts(name):
    clean = name.removeprefix("./").rstrip("/")
    path = PurePosixPath(clean)
    if (not path.parts or path.is_absolute() or any(c in clean for c in ("\\", ":", "\x00"))
            or ".." in path.parts):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path.parts


def copy_member(source, destination, size, limit):
    if size <= 0 or size > limit:
        raise ValueError(f"Unexpected archive member size for {destination.name}: {size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Duplicate CIFAR member: {destination.name}")
    with destination.open("xb") as handle:
        shutil.copyfileobj(source, handle)


def inspect_bundle(path, stage, classes):
    archives = {cls.filename for cls in classes}
    batches = official_members(classes)

    def handle(name, size, open_member):
        if name in (".", "./"):
            return
        parts = safe_parts(name)
        tail = "/".join(parts[-2:])
        if parts[-1] in archives:
            with open_member() as source:
                copy_member(source, stage / parts[-1], size, SIZES[parts[-1]])
        elif tail in batches:
            with open_member() as source:
                copy_member(source, stage / tail, size, 160_000_000)
        elif parts[-1] == "input_manifest.json":
            with open_member() as source:
                copy_member(source, stage / parts[-1], size, 1_000_000)
        elif parts[-1] == "reference_split.json":
            with open_member() as source:
                copy_member(source, stage / parts[-1], size, 4_000_000)

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    safe_parts(entry.filename)
                else:
                    handle(entry.filename, entry.file_size, lambda: archive.open(entry))
    else:
        with tarfile.open(path, mode="r:gz") as archive:
            expanded = 0
            for entry in archive:
                expanded += entry.size
                if expanded > 800_000_000:
                    raise ValueError("Expanded data archive exceeds the CIFAR input limit")
                if entry.isdir():
                    if entry.name not in (".", "./"):
                        safe_parts(entry.name)
                elif entry.isfile():
                    handle(entry.name, entry.size, lambda: archive.extractfile(entry))
                else:
                    raise ValueError(f"Unsupported TAR member: {entry.name!r}")


def complete(root, classes, boundary):
    for cls in classes:
        archive = root / cls.filename
        members = [root / cls.base_folder / name for name, _ in cls.train_list]
        if not ((archive.is_file() and archive.resolve().is_relative_to(boundary)) or
                all(p.is_file() and p.resolve().is_relative_to(boundary) for p in members)):
            return False
    return True


def validate_declared(root, actual, boundary):
    path = root / "input_manifest.json"
    if not path.exists():
        return
    if not path.resolve().is_relative_to(boundary):
        raise ValueError("Dataset checksum manifest escapes the selected input")
    declared = read_json(path)["files"]
    for name, record in actual.items():
        if name in declared and declared[name] != record:
            raise ValueError(f"Dataset manifest does not match {name}")


def discover(mount, relative, cache, *, use_cifar100=True):
    chosen = selected_folder(Path(mount), relative)
    boundary = chosen if chosen.is_dir() else chosen.parent
    classes = required_sources(use_cifar100)
    if cache.is_dir():
        records = validate_tree(cache, classes)
        validate_declared(cache, records, cache)
        return cache, records
    folders = [chosen] if chosen.is_dir() else []
    frontier = folders[:]
    for _ in range(3):
        more = sorted({p.resolve() for folder in frontier for p in folder.iterdir()
                       if p.is_dir() and not p.name.startswith(".") and p.resolve().is_relative_to(boundary)}
                      - set(folders))
        folders.extend(more)
        frontier = more
    direct = [f for f in folders if complete(f, classes, boundary)]
    if len(direct) > 1:
        raise ValueError("Multiple complete CIFAR roots; select one using dataset_subdir")
    if direct:
        records = validate_tree(direct[0], classes)
        validate_declared(direct[0], records, boundary)
        return direct[0], records

    files = ([chosen] if chosen.is_file() else
             sorted({p.resolve() for f in folders for p in f.iterdir()
                     if p.is_file() and p.resolve().is_relative_to(boundary)}))
    stage = make_stage(cache.parent)
    ordinary = {cls.filename: cls.tgz_md5 for cls in classes}
    for path in files:
        if path.name in ordinary and not (stage / path.name).exists():
            if md5_file(path) != ordinary[path.name]:
                raise ValueError(f"Official archive checksum failed: {path}")
            shutil.copyfile(path, stage / path.name)
    if not complete(stage, classes, stage):
        for cls in classes:
            for name, _ in cls.train_list:
                member = next((folder / name for folder in folders
                               if folder.name == cls.base_folder and (folder / name).is_file()
                               and (folder / name).resolve().is_relative_to(boundary)), None)
                if member is not None:
                    target = stage / cls.base_folder / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(member, target)
    if not complete(stage, classes, stage):
        bundles = [p for p in files if p.name not in SIZES and p.stat().st_size <= 420_000_000
                   and (p.suffix.lower() == ".zip" or p.name.lower().endswith(".tar.gz"))]
        if len(bundles) != 1:
            raise ValueError("Select one CIFAR bundle or the directory containing both official sources")
        inspect_bundle(bundles[0], stage, classes)
    records = validate_tree(stage, classes)
    validate_declared(stage, records, stage)
    commit_stage(stage, cache)
    return cache, records


def bundle(raw, reference, output):
    records = verify_archives(raw)
    reference_data = read_json(reference)
    from core.splits import validate_manifest
    validate_manifest(reference_data, 50_000, 50_000, 10_000)
    if output.suffix != ".zip":
        raise ValueError("Upload package must be .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_STORED) as archive:
        for name in records:
            archive.write(raw / name, name)
        archive.writestr("input_manifest.json", json.dumps({"files": records}, indent=2))
        archive.write(reference, "reference_split.json")
    with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{file_hash(output)}  {output.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle(args.data_dir, args.reference_split, args.out)
