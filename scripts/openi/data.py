"""Prepare or verify the two official CIFAR archives needed by training."""
import argparse
from pathlib import Path
import tempfile
from urllib.request import urlopen
import shutil
from zipfile import ZIP_STORED, ZipFile

from core.data import md5_file
from core.runtime import file_hash
from scripts.openi.state import commit_stage, make_stage, write_json


FILES = {
    "cifar-10-python.tar.gz": "c58f30108f718f92721af3b95e74349a",
    "cifar-100-python.tar.gz": "eb9058c3a382ffc7106e4002c42a8d85",
}
SIZES = {"cifar-10-python.tar.gz": 170498071, "cifar-100-python.tar.gz": 169001437}
BUNDLE = "robusthfl-cifar-v1.zip"
SOURCES = ("https://cave.cs.toronto.edu/kriz/{name}",
           "https://zenodo.org/records/10089977/files/{name}")


def verify_archives(root):
    records = {}
    for name, expected in FILES.items():
        path = root / name
        if not path.is_file() or md5_file(path) != expected:
            raise ValueError(f"Missing or invalid official archive: {path}")
        records[name] = {"bytes": path.stat().st_size, "md5": expected, "sha256": file_hash(path)}
    return records


def make_bundle(root, path):
    records = verify_archives(root)
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(path)
    if path.name != BUNDLE:
        raise ValueError(f"Use the recognized upload name {BUNDLE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "x", compression=ZIP_STORED) as archive:
        for name in FILES:
            archive.write(root / name, arcname=name)
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{file_hash(path)}  {path.name}\n")
    return records


def locate_mounted_data(mount, subdir, run_root):
    mount = mount.resolve()
    chosen = (mount / subdir).resolve()
    if not chosen.is_relative_to(mount):
        raise ValueError("--dataset-subdir must remain inside the c2net dataset mount")
    folders = [chosen] if chosen.is_dir() else []
    for _ in range(2):
        next_level = {p.resolve() for folder in folders for p in folder.iterdir()
                      if p.is_dir() and not p.name.startswith(".") and p.resolve().is_relative_to(mount)}
        folders.extend(sorted(next_level - set(folders)))
    direct = [p for p in folders if all((p / name).is_file() and (p / name).resolve().is_relative_to(mount)
                                        for name in FILES)]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise ValueError("Multiple CIFAR archive pairs are mounted; set --dataset-subdir to one pair")
    bundles = ([chosen] if chosen.name == BUNDLE and chosen.is_file() else
               [p / BUNDLE for p in folders if (p / BUNDLE).is_file()])
    if len(bundles) != 1 or not bundles[0].resolve().is_relative_to(mount):
        raise ValueError(f"Expected one {BUNDLE} or one directory with both official archives under {chosen}")
    cache = run_root / ".cifar-input"
    if not cache.is_dir():
        staged = make_stage(run_root)
        with ZipFile(bundles[0]) as archive:
            sizes = {entry.filename: entry.file_size for entry in archive.infolist()}
            if len(archive.infolist()) != len(SIZES) or sizes != SIZES:
                raise ValueError(f"Unexpected file names or sizes in {BUNDLE}")
            for name in FILES:
                with archive.open(name) as src, (staged / name).open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        verify_archives(staged)
        commit_stage(staged, cache)
    return cache


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        if path.exists():
            if md5_file(path) != expected:
                raise ValueError(f"Refusing to overwrite existing file with incorrect MD5: {path}")
            continue
        errors = []
        for source in SOURCES:
            url = source.format(name=name)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix=f".{name}.", delete=False) as pending:
                    temporary = Path(pending.name)
                    with urlopen(url, timeout=30) as response:
                        shutil.copyfileobj(response, pending)
                if md5_file(temporary) != expected:
                    raise ValueError("download checksum mismatch")
                path.hardlink_to(temporary)
                break
            except (OSError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError("\n".join(errors))
    return verify_archives(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, help=f"Create the manual-upload ZIP named {BUNDLE}")
    args = parser.parse_args()
    if args.manifest.exists():
        parser.error("--manifest already exists")
    result = prepare(args.data_dir) if args.download else verify_archives(args.data_dir)
    if args.bundle:
        make_bundle(args.data_dir, args.bundle)
        result["upload_bundle"] = {"name": args.bundle.name, "bytes": args.bundle.stat().st_size,
                                   "sha256": file_hash(args.bundle)}
    write_json(args.manifest, result)


if __name__ == "__main__":
    main()
