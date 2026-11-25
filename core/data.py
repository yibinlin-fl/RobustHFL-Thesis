"""The training data API exposes only private train, local validation, and unlabeled proxy views."""
import hashlib
import io
import pickle
import shutil
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from core.corruptions import Corruption, PRIVATE_ROLES
from core.hashing import file_hash
from core.splits import digest, freeze, make_manifest


def md5_file(path):
    return file_hash(path, "md5")


def load_cifar(root, cls, *, download, final_test=False):
    if final_test and cls is not CIFAR10:
        raise ValueError("Only CIFAR-10 has a final test role")
    root = Path(root)
    records = cls.test_list if final_test else cls.train_list
    directory = root / cls.base_folder
    paths = [(directory / name, checksum) for name, checksum in records]
    extracted = all(path.is_file() and md5_file(path) == checksum for path, checksum in paths)
    if not extracted:
        archive = root / cls.filename
        if not archive.is_file() or md5_file(archive) != cls.tgz_md5:
            if not download:
                raise FileNotFoundError(f"Verified {cls.__name__} archive or official files required under {root}")
            root.mkdir(parents=True, exist_ok=True)
            pending = archive.with_suffix(".download")
            url = f"https://cave.cs.toronto.edu/kriz/{cls.filename}"
            print(f"Downloading {url} to {archive}", flush=True)
            with urlopen(url, timeout=30) as response, pending.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if md5_file(pending) != cls.tgz_md5:
                raise ValueError(f"Official archive checksum failed: {pending}")
            pending.replace(archive)
        # Read only the requested members. Training never unpickles the test batch.
        with tarfile.open(archive, "r:gz") as handle:
            wanted = {f"{cls.base_folder}/{name}": checksum for name, checksum in records}
            batches = {}
            for member in handle:
                name = member.name.removeprefix("./")
                if name in wanted:
                    raw = handle.extractfile(member).read()
                    if hashlib.md5(raw).hexdigest() != wanted[name]:
                        raise ValueError(f"CIFAR file checksum failed: {name}")
                    batches[name] = pickle.load(io.BytesIO(raw), encoding="latin1")
                if len(batches) == len(wanted):
                    break
            batches = [batches[f"{cls.base_folder}/{name}"] for name, _ in records]
    else:
        batches = []
        for path, _ in paths:
            with path.open("rb") as handle:
                batches.append(pickle.load(handle, encoding="latin1"))
    key = "labels" if cls is CIFAR10 else "fine_labels"
    images = np.concatenate([record["data"] for record in batches]).reshape(-1, 3, 32, 32)
    images.flags.writeable = False
    if cls is CIFAR100:
        return images, None
    labels = np.asarray([label for record in batches for label in record[key]], dtype=np.int64)
    labels.flags.writeable = False
    return images, labels


class ImageRole(Dataset):
    def __init__(self, images, labels, indices, role, seed, corruption=None, limit=None, *, parts=None):
        if role not in PRIVATE_ROLES | {"in_domain_proxy", "cifar100_proxy"}:
            raise ValueError(f"Role unavailable in training API: {role}")
        if role in PRIVATE_ROLES and labels is None:
            raise ValueError("Private views require labels")
        if role not in PRIVATE_ROLES and (labels is not None or corruption is not None or parts):
            raise ValueError("Proxy views are unlabeled and uncorrupted")
        rng = np.random.default_rng(seed)
        # Randomize first so smoke subsets and corruption membership are reproducible.
        ids = rng.permutation(indices)
        self.ids = ids[:limit].tolist()
        if not self.ids:
            raise ValueError("Dataset view is empty")
        self.images = images[self.ids]
        self.images.flags.writeable = False
        self.labels = labels[self.ids] if labels is not None else None
        if self.labels is not None:
            self.labels.flags.writeable = False
        self.role = role
        self.seed = seed
        self.corruption = corruption
        if parts is None:
            count = round(len(self.ids) * corruption.ratio) if corruption else 0
            self.selected = set(rng.choice(self.ids, count, replace=False).tolist())
            self.assignments = dict.fromkeys(self.selected, corruption)
        else:
            if sum(p.ratio for p in parts) > 1:
                raise ValueError("Corruption mixture exceeds 100%")
            order = rng.permutation(self.ids)
            self.assignments = {}
            ratio = 0
            start = 0
            for part in parts:
                ratio += part.ratio
                stop = round(len(self.ids) * ratio)
                self.assignments.update(dict.fromkeys(order[start:stop].tolist(), part))
                start = stop
            self.selected = set(self.assignments)
        self._positions = {idx: position for position, idx in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        idx = self.ids[position]
        image = self.images[position].astype(np.float32) / 255
        if idx in self.selected:
            image = self.assignments[idx].apply(image, role=self.role, seed=[self.seed, idx])
        x = torch.from_numpy(image)
        if self.labels is None:
            return x
        return x, int(self.labels[position]), idx

    def position(self, idx):
        return self._positions[idx]

    def label_for(self, idx):
        return int(self.labels[self.position(idx)])


@dataclass(frozen=True)
class Data:
    train: list
    val_clean: list
    val_mixed: list
    val_robust: list
    proxies: dict
    corruptions: list
    split_hash: str
    view_hash: str


def read_train_sources(cfg):
    root = Path(cfg["data_root"])
    x10, y10 = load_cifar(root, CIFAR10, download=cfg["download"])
    x100, _ = (load_cifar(root, CIFAR100, download=cfg["download"])
               if cfg["split"]["proxy100_size"] else (None, None))
    if len(x10) != 50_000 or (x100 is not None and len(x100) != 50_000):
        raise ValueError("Unexpected official CIFAR train sizes")
    return x10, y10, x100


def setup_data(cfg, sources=None):
    x10, y10, x100 = read_train_sources(cfg) if sources is None else sources
    split = make_manifest(y10, len(x100) if x100 is not None else 0, 10_000,
                          seed=cfg["seed"], clients=len(cfg["clients"]), **cfg["split"])
    split_hash = freeze(cfg["manifest"], split)
    limit = cfg.get("limits", {})
    train, val_clean, val_mixed, val_robust, specs = [], [], [], [], []
    for i, client in enumerate(cfg["clients"]):
        spec = Corruption(**client["corruption"])
        parts = [Corruption(**p) for p in client["corruption_parts"]] if "corruption_parts" in client else None
        specs.append(spec)
        train.append(ImageRole(x10, y10, split["private_train"][i]["indices"],
                               "private_train", cfg["seed"] + 10*i + 1, spec, limit.get("train"), parts=parts))
        args = (x10, y10, split["client_local_val"][i]["indices"], "client_local_val", cfg["seed"] + 10*i + 2)
        val_clean.append(ImageRole(*args, limit=limit.get("val")))
        val_mixed.append(ImageRole(*args, corruption=spec, limit=limit.get("val"), parts=parts))
        val_robust.append(ImageRole(*args, corruption=replace(spec, ratio=1), limit=limit.get("val")))
    proxies = {
        role: ImageRole(images, None, split[role]["indices"], role, cfg["seed"] + 3, limit=limit.get("proxy"))
        for role, images in [("in_domain_proxy", x10), ("cifar100_proxy", x100)]
        if images is not None
    }
    views = {
        "split_hash": split_hash,
        "train": [[d.ids, sorted(d.selected), vars(d.corruption)] for d in train],
        "val": [[d.ids, sorted(d.selected), vars(d.corruption)] for d in val_mixed],
        "proxies": {role: ds.ids for role, ds in proxies.items()},
    }
    if any("corruption_parts" in client for client in cfg["clients"]):
        views["mixtures"] = [
            [[idx, vars(spec)] for idx, spec in sorted(dataset.assignments.items())]
            for dataset in train + val_mixed
        ]
    return Data(train, val_clean, val_mixed, val_robust, proxies, specs, split_hash, digest(views))
