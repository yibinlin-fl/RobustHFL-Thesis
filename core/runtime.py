"""Config, deterministic sampling and shared optimizer/checkpoint primitives."""
import argparse
from contextvars import ContextVar
import hashlib
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from core.models import build_model


STEP_DEADLINE = ContextVar("step_deadline", default=None)


def config(path, **overrides):
    path = Path(path)
    current = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = current.pop("extends", None)
    if base:
        current = merge(config(path.parent / base), current)
    return {**current, **{k: v for k, v in overrides.items() if v is not None}}


def merge(base, overrides):
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def cli(description, default):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def read_args(args):
    return config(args.config, data_root=args.data_root, manifest=args.manifest, out=args.out, device=args.device)


def initialize(cfg):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(cfg.get("threads", 2))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable (torch={torch.__version__}, build={torch.version.cuda})")
    return torch.device(device)


def sample_batches(dataset, size, steps, seed):
    if size < 1 or steps < 0 or len(dataset) < size:
        raise ValueError("Invalid batch size or step budget for dataset")
    rng = np.random.default_rng(seed)
    done = 0
    while done < steps:
        positions = rng.permutation(len(dataset))
        for batch in positions[:len(positions) // size * size].reshape(-1, size):
            yield batch.tolist()
            done += 1
            if done == steps:
                break


def loader(dataset, size):
    return DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)


def batch_at(dataset, positions, device):
    records = [dataset[i] for i in positions]
    if isinstance(records[0], torch.Tensor):
        return torch.stack(records).to(device)
    x, y, ids = zip(*records)
    return torch.stack(x).to(device), torch.tensor(y, device=device), torch.tensor(ids)


def normalize(x):
    # Shared published training-channel constants apply after raw-pixel operations.
    mean = x.new_tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]
    std = x.new_tensor([0.2470, 0.2435, 0.2616])[None, :, None, None]
    return (x - mean) / std


def optimizer(model, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"])


def step(loss, model, opt, clip):
    guard = STEP_DEADLINE.get()
    if guard is not None:
        guard()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    opt.step()


def init_client(name, cfg, device, index):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"] + index)
        model = build_model(name).to(device)
    return model, optimizer(model, cfg)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model_state_hash(model):
    checksum = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        checksum.update(f"{name}|{array.dtype}|{array.shape}\n".encode())
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def save_checkpoint(path, model, opt, cfg, data, index):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.pt")
    payload = {
        "version": 1, "seed": cfg["seed"], "client": index,
        "model_name": cfg["clients"][index]["model"], "split_hash": data.split_hash,
        "view_hash": data.view_hash, "model": model.state_dict(), "optimizer": opt.state_dict(),
    }
    with temp.open("wb") as handle:
        torch.save(payload, handle)
    checksum = file_hash(temp)
    temp.replace(path)
    return checksum


def read_checkpoint(path, cfg, device, index, split_hash, view_hash=None):
    record = torch.load(path, map_location=device, weights_only=True)
    expected = (1, cfg["seed"], index, cfg["clients"][index]["model"], split_hash)
    actual = tuple(record[k] for k in ("version", "seed", "client", "model_name", "split_hash"))
    if actual != expected or (view_hash is not None and record["view_hash"] != view_hash):
        raise ValueError(f"Checkpoint provenance does not match protocol: {path}")
    model, opt = init_client(record["model_name"], cfg, device, index)
    model.load_state_dict(record["model"])
    opt.load_state_dict(record["optimizer"])
    return model, opt
