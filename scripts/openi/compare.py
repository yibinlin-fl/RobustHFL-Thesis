"""Compute Phase 8 interaction from four completed arms of the same seed and protocol."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

from core.splits import digest
from scripts.openi.state import ShardLog, commit_stage, make_stage, read_json, write_json


def compare(paths, destination):
    if destination.exists():
        raise FileExistsError(destination)
    reference = None
    rows = {}
    for group, path in zip("ABCD", paths, strict=True):
        run = read_json(path / "run.json")
        status = read_json(path / "status.json")
        if status["execution_status"] != "complete" or status["stage"] != "abcd" or status["group"] != group:
            raise ValueError(f"Expected completed group {group}: {path}")
        identity = {key: value for key, value in run["identity"].items() if key != "group"}
        if reference is not None and identity != reference:
            raise ValueError("Groups differ in seed, source, runtime or scientific configuration")
        reference = identity
        with (path / "training" / "metrics" / "groups.csv").open(encoding="utf-8", newline="") as handle:
            record, = csv.DictReader(handle)
        if record["slice"] != group or record["run_id"] != run["run_id"]:
            raise ValueError("Group log identity does not match run metadata")
        rows[group] = record
    same = {tuple(row[k] for k in ("seed", "split_hash", "view_hash", "steps")) for row in rows.values()}
    if len(same) != 1:
        raise ValueError("Group split, view or step budgets do not match")
    seed, split_hash, view_hash, steps = same.pop()
    stage = make_stage(destination.parent)
    run_ids = {g: row["run_id"] for g, row in rows.items()}
    log = ShardLog(stage, digest(run_ids), {"seed": int(seed)},
                   SimpleNamespace(split_hash=split_hash, view_hash=view_hash))
    ckpt_hash = digest({g: row["ckpt_hash"] for g, row in rows.items()})
    for metric in ("clean_acc", "mixed_acc", "corrupt_acc", "worst_client_acc"):
        a, b, c, d = (float(rows[g][metric]) for g in "ABCD")
        log.row("interaction", slice_name="ABCD", ckpt_hash=ckpt_hash,
                metric=metric, A=a, B=b, C=c, D=d, interaction=(d-b)-(c-a), matched_steps=int(steps))
    write_json(stage / "comparison.json", {"run_ids": run_ids, "protocol": reference})
    commit_stage(stage, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for group in "ABCD":
        parser.add_argument(f"--{group.lower()}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare([args.a, args.b, args.c, args.d], args.out)


if __name__ == "__main__":
    main()
