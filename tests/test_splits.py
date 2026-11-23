import numpy as np
import pytest

from core.corruptions import Corruption
from core.data import ImageRole
from core.splits import digest, freeze, make_manifest, validate_manifest


@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_disjoint_source_qualified_frozen_splits(partition, tmp_path):
    y = np.arange(500) % 10
    opts = dict(seed=2026, clients=4, partition=partition, alpha=.3,
                proxy10_size=30, proxy100_size=25, val_fraction=.2)
    manifest = make_manifest(y, 500, 100, **opts)
    assert manifest == make_manifest(y, 500, 100, **opts)
    validate_manifest(manifest, 500, 500, 100)
    path = tmp_path / "splits_manifest.json"
    assert freeze(path, manifest) == freeze(path, manifest)
    data = manifest["client_local_val"][0]
    data["indices"][0] = manifest["private_train"][0]["indices"][0]
    data["sha256"] = digest([data["source"], data["indices"]])
    manifest["split_hash"] = digest({k: v for k, v in manifest.items() if k != "split_hash"})
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(manifest, 500, 500, 100)
    with pytest.raises(ValueError, match="Frozen"):
        freeze(path, manifest)


@pytest.mark.parametrize("name", ["gaussian_noise", "defocus_blur"])
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_corruption_range_and_source_immutability(name, severity):
    source = np.random.default_rng(10).integers(0, 256, (10, 3, 32, 32), dtype=np.uint8)
    saved = source.copy()
    spec = Corruption(name, severity, .6)
    ds = ImageRole(source, np.arange(10), range(10), "private_train", 23, spec)
    assert len(ds.selected) == 6
    for i in range(len(ds)):
        image, _, idx = ds[i]
        assert image.isfinite().all() and image.min() >= 0 and image.max() <= 1
        assert (image == ds[i][0]).all()
        if idx not in ds.selected:
            np.testing.assert_allclose(image.numpy(), source[idx]/255, atol=1e-7)
    assert np.array_equal(saved, source)
    with pytest.raises(ValueError, match="forbidden"):
        spec.apply(saved[0].astype(np.float32)/255, role="official_test", seed=0)
    with pytest.raises(ValueError, match="unlabeled"):
        ImageRole(source, np.arange(10), range(10), "in_domain_proxy", 23)
    with pytest.raises(ValueError, match="unavailable"):
        ImageRole(source, np.arange(10), range(10), "official_test", 23)
