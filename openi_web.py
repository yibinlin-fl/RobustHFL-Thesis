"""Startup file for a managed OpenI training job using an uploaded dataset."""
import argparse
from pathlib import Path
from tempfile import gettempdir

from scripts.openi.launch import PROFILE, main as launch


def web_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--stage", choices=PROFILE, default="smoke")
    parser.add_argument("--dataset_subdir", default=".")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--group", choices=tuple("ABCD"))
    parser.add_argument("--git_commit")
    parser.add_argument("--recovery_file")
    args = parser.parse_args(argv)
    if (args.stage == "abcd") != bool(args.group):
        parser.error("--group is required only for abcd")
    flags = [
        "--run-id", args.run_id, "--stage", args.stage,
        "--runs-root", str(Path(gettempdir()) / "robusthfl-openi"),
        "--c2net", "--require-v100", "--dataset-subdir", args.dataset_subdir,
    ]
    if args.stage != "smoke":
        flags.append("--calibrate-batch")
    if args.recovery_file:
        flags.extend(("--resume", "--recovery-file", args.recovery_file))
    for key in ("seed", "group", "git_commit"):
        value = getattr(args, key)
        if value is not None:
            flags.extend((f"--{key.replace('_', '-')}", str(value)))
    return flags


if __name__ == "__main__":
    launch(web_arguments())
