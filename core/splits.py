"""Frozen source-qualified CIFAR splits. Indices are zero-based within each source."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_manifest(labels, n100, ntest, *, seed, clients, partition, alpha,
                  proxy10_size, proxy100_size, val_fraction, min_private=2):
    labels = np.asarray(labels)
    if (clients < 2 or not 0 < val_fraction < 1 or not 0 < proxy10_size < len(labels)
            or not (0 < proxy100_size <= n100 or proxy100_size == n100 == 0) or min_private < 2
            or partition not in {"iid", "dirichlet"} or alpha <= 0):
        raise ValueError("Invalid split configuration")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(labels))
    proxy10 = sorted(shuffled[:proxy10_size].tolist())
    population = shuffled[proxy10_size:]
    if len(population) < clients * (min_private + 1):
        raise ValueError("Insufficient CIFAR-10 training population")
    if partition == "iid":
        groups = np.array_split(population, clients)
    else:
        for _ in range(200):
            groups = [[] for _ in range(clients)]
            for label in np.unique(labels):
                items = population[labels[population] == label]
                sizes = rng.multinomial(len(items), rng.dirichlet(np.full(clients, alpha)))
                for group, chunk in zip(groups, np.split(items, np.cumsum(sizes)[:-1])):
                    group.extend(chunk.tolist())
            if min(map(len, groups)) >= min_private + 1:
                break
        else:
            raise ValueError("Dirichlet split failed minimum private quota; increase alpha or lower quota")

    def entry(source, indices):
        return {"source": source, "indices": indices, "sha256": digest([source, indices])}

    private, local_val = [], []
    for group in groups:
        group = rng.permutation(group)
        nval = min(len(group) - min_private, max(1, round(len(group) * val_fraction)))
        private.append(entry("cifar10/train", sorted(group[nval:].tolist())))
        local_val.append(entry("cifar10/train", sorted(group[:nval].tolist())))
    manifest = {
        "version": 1, "seed": seed, "partition": partition, "alpha": alpha,
        "val_fraction": val_fraction, "min_private": min_private,
        "cifar10_train_labels_sha256": digest(labels.tolist()),
        "private_train": private, "client_local_val": local_val,
        "in_domain_proxy": entry("cifar10/train", proxy10),
        "cifar100_proxy": entry("cifar100/train", sorted(rng.choice(n100, proxy100_size, replace=False).tolist())),
        "official_test": entry("cifar10/test", list(range(ntest))),
    }
    manifest["split_hash"] = digest(manifest)
    validate_manifest(manifest, len(labels), n100, ntest)
    return manifest


def validate_manifest(manifest, n10, n100, ntest):
    sizes = {"cifar10/train": n10, "cifar100/train": n100, "cifar10/test": ntest}
    private, local_val = manifest["private_train"], manifest["client_local_val"]
    if len(private) < 2 or len(private) != len(local_val):
        raise ValueError("Mismatched client split counts")
    expected = [(r, "cifar10/train") for r in private + local_val]
    expected.extend((manifest[key], source) for key, source in
                    [("in_domain_proxy", "cifar10/train"), ("cifar100_proxy", "cifar100/train"),
                     ("official_test", "cifar10/test")])
    sets = []
    for record, source in expected:
        indices = record["indices"]
        ids = set(indices)
        optional_empty = source == "cifar100/train" and n100 == 0
        if (record["source"] != source or (not ids and not optional_empty) or len(ids) != len(indices)
                or any(type(i) is not int or not 0 <= i < sizes[source] for i in ids)):
            raise ValueError("Invalid source-qualified indices")
        if record["sha256"] != digest([source, indices]):
            raise ValueError("Invalid split entry SHA-256")
        sets.append({(source, i) for i in ids})
    if any(a & b for a, b in combinations(sets, 2)):
        raise ValueError("Split overlap")
    if sum(len(s) for s in sets[:-2]) != n10 or len(sets[-1]) != ntest:
        raise ValueError("Incomplete CIFAR-10 coverage")
    if manifest["split_hash"] != digest({k: v for k, v in manifest.items() if k != "split_hash"}):
        raise ValueError("Invalid manifest SHA-256")


def freeze(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Frozen split mismatch: {path}")
    return manifest["split_hash"]


if __name__ == "__main__":
    from core.data import setup_data
    from core.runtime import config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(setup_data(config(args.config)).split_hash)
