"""Create an OpenI source-only ZIP from this checkout, without data or local results."""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from core.runtime import file_hash
from scripts.openi.state import REPO, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(REPO / "results") or output.is_relative_to(REPO / "data"):
        parser.error("Use an output path outside the checkout's results and data")
    if output.suffix != ".zip":
        parser.error("--out must be a .zip path")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as bundle:
        extra = [".gitignore", "README.md", "docs/research/OPENI_V100_SETUP.md", "docs/research/OPENI_DEPLOY_GUIDE.md"]
        for name in sorted(set(identity["source_files"]) | set(extra)):
            path = REPO / name
            bundle.writestr(name, path.read_bytes().replace(b"\r\n", b"\n"))
        bundle.writestr("SOURCE.json", json.dumps(identity, indent=2, sort_keys=True) + "\n")
    with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{file_hash(output)}  {output.name}\n")
    print(f"{output} {identity['source_sha256']}", flush=True)


if __name__ == "__main__":
    main()
