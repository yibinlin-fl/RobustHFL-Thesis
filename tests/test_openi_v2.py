from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from core.corruptions import Corruption
from core.data import ImageRole, md5_file
from core.hashing import stream_hash
from core.splits import make_manifest, validate_manifest
from scripts import openi_entry
from scripts.openi import v2_data
from scripts.openi.precheck_protocol import BLUR, GAUSSIAN, clients_for, qualify
from scripts.openi.precheck_job import MODES, affordable_prefix, csv_check


def test_optional_cifar100_still_covers_all_train_indices():
    manifest = make_manifest(np.repeat(range(10), 30), 0, 100, seed=1, clients=4,
                             partition="iid", alpha=.5, proxy10_size=32, proxy100_size=0, val_fraction=.1)
    assert manifest["cifar100_proxy"]["indices"] == []
    validate_manifest(manifest, 300, 0, 100)


def test_corruption_mixture_crosses_both_architectures_and_matches_global_counts():
    cfg = {"moderate": 3, "severe": 5, "moderate_fraction": .3, "severe_fraction": .3}
    images = np.zeros((100, 3, 32, 32), dtype=np.uint8)
    labels = np.arange(100) % 10
    compositions = []
    for scenario in ("heterogeneous", "uniform_mixture"):
        clients = clients_for(scenario, cfg)
        if scenario == "heterogeneous":
            assert {(c["model"], c["corruption"]["kind"]) for c in clients} == {
                (m, k) for m in ("ResNet10", "MobileNetV2") for k in (GAUSSIAN, BLUR)}
        counts = Counter()
        for i, c in enumerate(clients):
            parts = [Corruption(**p) for p in c["corruption_parts"]]
            ds = ImageRole(images, labels, np.arange(100), "private_train", i, parts=parts)
            counts.update((s.kind, s.severity) for s in ds.assignments.values())
            assert len(ds.selected) == 60
        compositions.append(counts)
    assert compositions[0] == compositions[1]
    assert not images.any()


def test_targz_input_accepts_safe_official_members_only(tmp_path, monkeypatch):
    class Fixture:
        filename = "test.tar.gz"
        base_folder = "input"
        train_list = ()
    monkeypatch.setattr(v2_data, "SIZES", {"test.tar.gz": 10})
    bundle = tmp_path / "bundle.tar.gz"
    stage = tmp_path / "stage"
    stage.mkdir()
    with tarfile.open(bundle, "w:gz") as tar:
        for name, contents in [("inner/test.tar.gz", b"official"), ("ignored.txt", b"not copied")]:
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))
    v2_data.inspect_bundle(bundle, stage, (Fixture,))
    assert (stage / "test.tar.gz").read_bytes() == b"official"
    assert not (stage / "ignored.txt").exists()
    with tarfile.open(bundle, "w:gz") as tar:
        info = tarfile.TarInfo("../test.tar.gz")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"a"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        v2_data.inspect_bundle(bundle, stage, (Fixture,))


def test_classification_gate_rejects_collapsed_checkpoint():
    data = SimpleNamespace(val_clean=[TensorDataset(torch.zeros(10, 3, 32, 32), torch.arange(10))])
    class Collapsed(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(len(x), 10)
    cfg = {"eval_batch_size": 4, "quality": {"min_clean_accuracy": .2, "majority_margin": .05,
           "min_predicted_classes": 3, "max_majority_prediction": .85}}
    assert not qualify([Collapsed()], data, cfg, torch.device("cpu"))[0]["passed"]


def test_c2net_prepare_then_error_still_attempts_upload(tmp_path, monkeypatch):
    from scripts.openi import precheck_job
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    repo = Path(openi_entry.__file__).resolve().parents[1]
    context = SimpleNamespace(code_path=repo, dataset_path=tmp_path, output_path=tmp_path / "output")
    def fail(*args):
        raise RuntimeError("sentinel")
    calls = []
    def upload():
        files = list(Path(context.output_path).rglob("failure.json"))
        assert len(files) == 1
        calls.append(json.loads(files[0].read_text()))
    monkeypatch.setattr(openi_entry, "published_upload", lambda: (context, upload))
    monkeypatch.setattr(v2_data, "discover", lambda *a, **k: (tmp_path, {}))
    monkeypatch.setattr(precheck_job, "v2_job", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        openi_entry.run(openi_entry.arguments(["--mode", "work1_precheck", "--run-id", "failure-test"]))
    assert calls[0]["status"] == "failed"
    root = Path(context.output_path) / "RobustHFL-V2" / "failure-test"
    assert "RuntimeError: sentinel" in (root / "execution.log").read_text()
    assert openi_entry.verified_files(root)


def test_streaming_checksums_without_hashlib_file_digest(tmp_path, monkeypatch):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    payload = b"CIFAR\x00" * 200_000
    path = tmp_path / "payload"
    path.write_bytes(payload)
    assert md5_file(path) == hashlib.md5(payload).hexdigest()
    expected = hashlib.sha256(payload).hexdigest()
    assert openi_entry.file_sha(path) == expected
    assert stream_hash(io.BytesIO(payload)) == expected


def test_csv_validation_rejects_non_finite(tmp_path):
    table = tmp_path / "invalid.csv"
    table.write_text("score,delta\nfin,NaN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        csv_check(table)


def test_budget_priority_keeps_work1_before_other_experiments():
    from core.runtime import config
    from scripts.openi.precheck_protocol import training_config
    repo = Path(openi_entry.__file__).resolve().parents[1]
    cfg = training_config(config(repo / "configs/openi_precheck.yaml"), repo, repo / "data", "heterogeneous")
    timing = {"sample_local_steps": 4, "sample_local_wall_seconds": .48,
              "sample_kd_steps": 4, "sample_kd_wall_seconds": .2,
              "sample_validation_wall_seconds": .4, "sample_validation_images_per_client": 64,
              "sample_group_state_tensor_bytes": 100}
    selected, _ = affordable_prefix(cfg, MODES["precheck_all"], True, timing, 7200)
    assert selected == ("work1",)
    with pytest.raises(TimeoutError):
        affordable_prefix(cfg, MODES["precheck_all"], True, timing, 1)


def test_upload_false_does_not_report_success(tmp_path):
    with pytest.raises(RuntimeError, match="returned False"):
        openi_entry.publish(tmp_path, lambda: False)
    status = json.loads((tmp_path / "upload_status.json").read_text())
    assert status["result"] == "failed" and not status["remote_download_verified"]
