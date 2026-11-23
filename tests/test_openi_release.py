from pathlib import Path
import json

import numpy as np
import pytest

from core.splits import make_manifest
from scripts.openi_entry import arguments, file_sha, restore_directory, verified_files
from scripts.openi.transfer_v2 import a100_plan, pack_recovery, unpack_recovery


def write(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")


def hash_files(root):
    write(root / "file_hashes.json", {"audit": "complete", "files": {
        p.relative_to(root).as_posix(): {"sha256": file_sha(p), "bytes": p.stat().st_size}
        for p in root.rglob("*") if p.is_file() and p.name != "file_hashes.json"
    }})


def test_downloaded_recovery_zip_can_start_in_independent_output(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    write(previous / "run_manifest.json", {"run_id": "work1-01", "purpose": "research"})
    saved = previous / "checkpoints" / "base_2x2" / "rounds"
    for round_id in ("000002", "000003"):
        state = saved / round_id / "state.pt"
        state.parent.mkdir(parents=True)
        state.write_bytes(round_id.encode())
    hash_files(previous)
    archive = tmp_path / "attached-model" / "recovery.zip"
    assert pack_recovery(previous, archive) == "work1-01"
    fresh = tmp_path / "next-task"
    fresh.mkdir()
    scratch = unpack_recovery(archive.parent, fresh / ".unpack")
    destination = fresh / "work1-01"
    restore_directory(scratch, destination, promote=True)
    assert len(verified_files(destination)) == 3
    assert (destination / "checkpoints/base_2x2/rounds/000002/state.pt").read_bytes() == b"000002"
    args = arguments(["--mode", "work2_frequency", "--run_id", "work1-01",
                      "--restore_mount", "model", "--restore_subdir", "recovery.zip"])
    assert args.resume and args.restore_mount == "model"
    with pytest.raises(FileExistsError):
        restore_directory(destination, destination)


def test_a100_plan_prefers_one_complete_mode_and_rejects_rtx(tmp_path):
    splits = make_manifest(np.arange(50_000) % 10, 50_000, 10_000, seed=1, clients=4, partition="iid",
                           alpha=.5, proxy10_size=128, proxy100_size=128, val_fraction=.1)
    write(tmp_path / "splits_manifest.json", splits)
    write(tmp_path / "run_manifest.json", {"run_id": "env-01", "env_check_only": True,
          "execution_status": "complete", "last_requested_mode": "env_check", "formal_optimizer_updates": 0,
          "checkpoint_count": 0, "split_hash": splits["split_hash"], "batch_size": 32,
          "identity": {"git_commit": "a"*40}})
    env = {"cuda_kernel_verified": True, "runtime": {"gpu": "NVIDIA A100"},
           "nvidia_smi": "NVIDIA A100", "deployment": {"managed": True, "c2net_prepare": True}}
    write(tmp_path / "environment.json", env)
    names = ("work1_precheck", "work2_heterogeneity", "work2_frequency", "precheck_all")
    write(tmp_path / "budget_estimate.json", {"sample_forwards": [{"shape": [32, 10]}]*4,
          "sample_data_images_per_second": 1., "official_train_read_images_per_second": 1.,
          "sample_local_wall_seconds": 1., "sample_kd_wall_seconds": 1.,
          "by_mode": {name: {"estimated_compute_wall_seconds": seconds, "estimated_updates": {"steps": 500},
                             "estimated_checkpoint_storage_bytes": 100}
                      for name, seconds in zip(names, (6400, 6900, 2500, 12000), strict=True)}})
    hash_files(tmp_path)
    limits = dict(max_seconds=7200, max_points=18, points_per_hour=9, reserve_seconds=900, commit="a"*40)
    plan = a100_plan(tmp_path, **limits)
    assert plan["recommended_single_mode"] == "work2_frequency"
    assert not plan["modes"]["work1_precheck"]["fits"]
    env["runtime"]["gpu"] = "NVIDIA RTX 3050"
    env["nvidia_smi"] = env["runtime"]["gpu"]
    write(tmp_path / "environment.json", env)
    hash_files(tmp_path)
    with pytest.raises(ValueError, match="No verified managed A100"):
        a100_plan(tmp_path, **limits)
