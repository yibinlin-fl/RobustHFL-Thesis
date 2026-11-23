import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.openi.artifacts import pack, safe_name, verify
from scripts.openi import data as openi_data
from scripts.openi.launch import parse, restore_mounted_run
from scripts.openi.state import round_folders
from openi_web import web_arguments


def test_recovery_package_keeps_log_shards_and_only_latest_state(tmp_path):
    root = tmp_path / "run"
    for i in (1, 2):
        round_dir = root / "training" / "rounds" / f"{i:06d}"
        round_dir.mkdir(parents=True)
        (round_dir / "state.pt").write_bytes(f"round {i}".encode())
        (round_dir / "rounds.csv").write_text(f"round\n{i}\n", encoding="utf-8")
    (root / ".lock").write_bytes(b"")
    archive = pack(root, tmp_path / "archive.tar.gz")
    restored = tmp_path / "restored"
    result = verify(archive, restored)
    assert len(result["files"]) == 3
    first, last = round_folders(restored / "training" / "rounds")
    assert (first / "rounds.csv").is_file()
    assert not (first / "state.pt").exists()
    assert (last / "state.pt").read_bytes() == b"round 2"
    with pytest.raises(FileExistsError):
        verify(archive, restored)
    with archive.open("ab") as handle:
        handle.write(b"tampering")
    with pytest.raises(ValueError, match="Outer archive"):
        verify(archive)


@pytest.mark.parametrize("name", ["../state.pt", "/state.pt", "nested/../state.pt", "C:/state.pt", r"dir\state.pt"])
def test_unsafe_archive_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_name(name)


def test_formal_launcher_requires_measured_batch_and_persistent_sync():
    base = ["--stage", "probe2", "--run-id", "p7-a", "--runs-root", "runs", "--data-dir", "data"]
    with pytest.raises(SystemExit):
        parse(base)
    with pytest.raises(SystemExit):
        parse(base + ["--calibration", "audit.json"])
    args = parse(base + ["--calibration", "audit.json", "--sync-dir", "sync"])
    assert args.calibration == Path("audit.json")
    assert args.seed is None and args.sync_every == 1


def test_web_startup_maps_form_fields_to_managed_job():
    smoke = parse(web_arguments(["--run_id", "web-smoke"]))
    formal = parse(web_arguments(["--run_id", "web-p7", "--stage", "probe2", "--seed", "17"]))
    assert smoke.c2net and smoke.require_v100 and smoke.stage == "smoke" and not smoke.calibrate_batch
    assert formal.c2net and formal.calibrate_batch and formal.seed == 17
    recovery = parse(web_arguments(["--run_id", "web-p7", "--stage", "probe2",
                                    "--recovery_file", "resume/state.tar.gz"]))
    assert recovery.resume and recovery.recovery_file == "resume/state.tar.gz"


def test_restore_from_mounted_recovery_archive(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text('{"run_id":"saved"}', encoding="utf-8")
    mount = tmp_path / "mount"
    pack(run, mount / "resume" / "state.tar.gz")
    context = SimpleNamespace(dataset_path=mount)
    destination = tmp_path / "restored"
    restore_mounted_run(context, "resume/state.tar.gz", destination, "saved")
    assert (destination / "run.json").is_file()
    with pytest.raises(ValueError, match="inside"):
        restore_mounted_run(context, "../state.tar.gz", tmp_path / "outside", "saved")


def test_mounted_zip_or_sdk_extraction(tmp_path, monkeypatch):
    content = {"cifar-10-python.tar.gz": b"test-10", "cifar-100-python.tar.gz": b"test-100"}
    monkeypatch.setattr(openi_data, "FILES", {k: hashlib.md5(v).hexdigest() for k, v in content.items()})
    monkeypatch.setattr(openi_data, "SIZES", {k: len(v) for k, v in content.items()})
    mount = tmp_path / "mount"
    extracted = mount / "uploaded-dataset" / "robusthfl-cifar-v1"
    extracted.mkdir(parents=True)
    for name, data in content.items():
        (extracted / name).write_bytes(data)
    assert openi_data.locate_mounted_data(mount, ".", tmp_path / "run") == extracted
    zip_mount = tmp_path / "zip-mounted"
    bundle = zip_mount / openi_data.BUNDLE
    openi_data.make_bundle(extracted, bundle)
    cached = openi_data.locate_mounted_data(zip_mount, ".", tmp_path / "run")
    assert {p.name: p.read_bytes() for p in cached.iterdir()} == content
    assert openi_data.locate_mounted_data(zip_mount, ".", tmp_path / "run") == cached
    with pytest.raises(ValueError, match="inside"):
        openi_data.locate_mounted_data(mount, "..", tmp_path / "run")
