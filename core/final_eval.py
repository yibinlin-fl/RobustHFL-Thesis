"""One-way final assessment of frozen checkpoints on official CIFAR-10 and CIFAR-10-C."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from core.data import load_cifar
from core.evaluator import accuracy
from core.logger import Budget, RunLog
from core.runtime import cli, file_hash, initialize, loader, read_args, read_checkpoint
from core.splits import digest, validate_manifest


CIFAR_C = ("brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
           "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
           "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur")


class FinalImages(Dataset):
    def __init__(self, images, labels, channels_last=False):
        self.images = images
        self.labels = labels
        self.channels_last = channels_last

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i].astype(np.float32) / 255
        return torch.from_numpy(image.transpose(2, 0, 1) if self.channels_last else image), int(self.labels[i])


def run(cfg, checkpoint_dir, cifar_c, kinds):
    device = initialize(cfg)
    manifest = json.loads(Path(cfg["manifest"]).read_text(encoding="utf-8"))
    validate_manifest(manifest, 50_000, 50_000 if manifest["cifar100_proxy"]["indices"] else 0, 10_000)
    if len(manifest["private_train"]) != len(cfg["clients"]):
        raise ValueError("Final-evaluation client count does not match split manifest")
    split_hash = manifest["split_hash"]
    checkpoint_dir = Path(checkpoint_dir)
    paths = [checkpoint_dir / f"client_{i}.pt" for i in range(len(cfg["clients"]))]
    hashes = [file_hash(path) for path in paths]
    view_hash = torch.load(paths[0], map_location="cpu", weights_only=True)["view_hash"]
    models = [read_checkpoint(path, cfg, device, i, split_hash, view_hash)[0] for i, path in enumerate(paths)]
    log = RunLog(cfg, SimpleNamespace(split_hash=split_hash, view_hash=view_hash), "official_final")
    log.json_event({"event": "official_inputs", "checkpoint_dir": str(checkpoint_dir),
                    "ckpt_hashes": hashes, "cifar_c": str(cifar_c), "corruptions": kinds})
    images, labels = load_cifar(cfg["data_root"], CIFAR10, download=cfg["download"], final_test=True)
    if len(labels) != 10_000:
        raise ValueError("Expected 10,000 official CIFAR-10 test labels")
    root = Path(cifar_c)
    labels_path = root / "labels.npy"
    corrupt_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if corrupt_labels.shape not in ((10_000,), (50_000,)) or not np.array_equal(
            corrupt_labels, np.tile(labels, corrupt_labels.size // 10_000)):
        raise ValueError("CIFAR-10-C labels differ from official CIFAR-10 test ordering")
    log.json_event({"event": "official_c_labels", "sha256": file_hash(labels_path)})
    corrupted = {}
    clean = []
    with Budget(device) as budget:
        for i, model in enumerate(models):
            clean.append(accuracy(model, loader(FinalImages(images, labels), cfg["eval_batch_size"]), device))
        for kind in kinds:
            path = root / f"{kind}.npy"
            all_images = np.load(path, mmap_mode="r", allow_pickle=False)
            if all_images.shape != (50_000, 32, 32, 3) or all_images.dtype != np.uint8:
                raise ValueError(f"Unexpected CIFAR-10-C shape or dtype: {path}")
            log.json_event({"event": "official_c_file", "type": kind, "sha256": file_hash(path)})
            scores = []
            for severity in range(1, 6):
                offset = (severity-1)*10_000
                ds = FinalImages(all_images[offset:offset+10_000], labels, channels_last=True)
                current = [accuracy(model, loader(ds, cfg["eval_batch_size"]), device) for model in models]
                for i, score in enumerate(current):
                    log.row("official_corrupt", slice_name=kind, ckpt_hash=hashes[i],
                            client=i, severity=severity, corrupt_acc=score)
                scores.append(current)
            corrupted[kind] = np.mean(scores, axis=0)
    by_type = {k: float(np.mean(v)) for k, v in corrupted.items()}
    client_robust = np.mean(list(corrupted.values()), axis=0)
    for i, (clean_acc, corrupt_acc) in enumerate(zip(clean, client_robust)):
        log.row("official_clients", slice_name="final", ckpt_hash=hashes[i], client=i,
                clean_acc=clean_acc, corrupt_acc=float(corrupt_acc))
    log.row("official_summary", slice_name="final", budget=budget.record(), ckpt_hash=digest(hashes),
            clean_acc=float(np.mean(clean)), corrupt_acc=float(np.mean(client_robust)),
            worst_client_acc=float(min(client_robust)), corrupt_acc_by_type=json.dumps(by_type, sort_keys=True))


if __name__ == "__main__":
    parser = cli(__doc__, "configs/base.yaml")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--cifar-c", required=True, help="Extracted official CIFAR-10-C directory")
    parser.add_argument("--corruptions", nargs="+", choices=CIFAR_C, default=list(CIFAR_C))
    args = parser.parse_args()
    cfg = read_args(args)
    if args.out is None:
        cfg["out"] = "results/runs/final"
    run(cfg, args.checkpoints, args.cifar_c, list(dict.fromkeys(args.corruptions)))
