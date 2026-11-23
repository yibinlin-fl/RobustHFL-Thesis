"""Client-local EMA prototypes, frozen diagnostic scores, and differentiable local losses."""
from dataclasses import dataclass, field
from math import log

import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from core.runtime import batch_at, loader, normalize, sample_batches, step


MODES = ("CE", "Fixed Consistency", "Random Downweight", "Loss Downweight", "Candidate Quality-Modulated")
SIGNALS = ("s_i", "u_i", "loss_i", "loss_slope_i", "rand_i")


def weak_view(x, seed):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return (x + 0.01 * noise).clamp(0, 1)


def js(p, q):
    midpoint = (p + q) / 2
    return ((F.kl_div(midpoint.clamp_min(1e-8).log(), p, reduction="none").sum(1)
             + F.kl_div(midpoint.clamp_min(1e-8).log(), q, reduction="none").sum(1)) / 2).clamp_min(0)


@dataclass
class LocalHistory:
    prototypes: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)


@dataclass
class Scores:
    ids: np.ndarray
    raw: dict
    percentile: dict
    quality: dict

    def weights(self, ids, mode, scale, device):
        if mode == "Fixed Consistency":
            return torch.ones(len(ids), device=device)
        key = {"Random Downweight": "rand_i", "Loss Downweight": "loss_i",
               "Candidate Quality-Modulated": "candidate"}[mode]
        return torch.tensor([self.quality[scale][key][idx] for idx in ids.tolist()],
                            device=device, dtype=torch.float32)


@torch.no_grad()
def observe(model, dataset, history, cfg, device, round_no, seed, *, score=True):
    if dataset.role != "private_train":
        raise ValueError("Local training signals require private_train")
    model.eval()
    raw = {name: [] for name in SIGNALS}
    if cfg.get("probe1", {}).get("include_entropy", False):
        raw["entropy_i"] = []
    ids_all = []
    class_sums = {}
    momentum = cfg["local"]["prototype_ema"]
    rng = np.random.default_rng(seed)
    for batch_no, (x, y, ids) in enumerate(loader(dataset, cfg["eval_batch_size"])):
        x, y = x.to(device), y.to(device)
        logits, features = model(normalize(x), return_feat=True)
        ce = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
        for label in y.unique().tolist():
            value = features[y == label].sum(0)
            class_sums[label] = class_sums.get(label, 0) + value
        if score:
            prior = torch.stack([history.prototypes.get(label, torch.zeros_like(features[0]))
                                 for label in y.tolist()])
            raw["s_i"].extend((features * prior).sum(1).cpu().tolist())
            shifted = model(normalize(weak_view(x, seed + batch_no)))
            raw["u_i"].extend(js(logits.softmax(1), shifted.softmax(1)).cpu().tolist())
            if "entropy_i" in raw:
                log_probs = logits.log_softmax(1)
                raw["entropy_i"].extend((-(log_probs.exp() * log_probs).sum(1)).cpu().tolist())
            raw["loss_i"].extend(ce.tolist())
            for idx, value in zip(ids.tolist(), ce):
                previous = history.losses.get(idx)
                slope = (float(value) - previous[1]) / (round_no - previous[0]) if previous else np.nan
                raw["loss_slope_i"].append(slope)
            raw["rand_i"].extend(rng.random(len(ids)).tolist())
        history.losses.update({idx: (round_no, float(value)) for idx, value in zip(ids.tolist(), ce)})
        ids_all.extend(ids.tolist())
    for label, total in class_sums.items():
        current = F.normalize(total, dim=0)
        previous = history.prototypes.get(label, current)
        history.prototypes[label] = F.normalize(momentum*previous + (1-momentum)*current, dim=0).detach()
    if not score:
        return None

    raw = {key: np.asarray(values) for key, values in raw.items()}
    percentile = {}
    for name, values in raw.items():
        valid = np.isfinite(values)
        ranks = np.full(len(values), np.nan)
        if valid.any():
            ranks[valid] = (rankdata(values[valid], method="average") - 0.5) / valid.sum()
        percentile[name] = ranks
    physical = {
        "s_i": (np.clip(raw["s_i"], -1, 1) + 1) / 2,
        "u_i": 1 - np.clip(raw["u_i"] / log(2), 0, 1),
        "loss_i": np.exp(-raw["loss_i"]),
        "rand_i": raw["rand_i"],
    }
    relative = {key: percentile[key] for key in ("s_i", "u_i", "loss_i", "rand_i")}
    relative["u_i"] = 1 - relative["u_i"]
    relative["loss_i"] = 1 - relative["loss_i"]
    quality = {}
    for scale, mapping in [("raw", physical), ("percentile", relative)]:
        mapping["candidate"] = (mapping["s_i"] + mapping["u_i"]) / 2
        quality[scale] = {key: dict(zip(ids_all, values)) for key, values in mapping.items()}
    return Scores(np.asarray(ids_all), raw, percentile, quality)


def local_loss(model, x, y, mode, local, seed, q=None):
    logits = model(normalize(x))
    ce = F.cross_entropy(logits, y, reduction="none")
    if mode == "CE":
        return ce.mean()
    if q is None:
        raise ValueError("Differentiated local loss requires detached sample quality")
    q = q.detach()
    a = local["a_min"] + (1 - local["a_min"]) * q
    rule = local["consistency_weight"]
    if rule == "const_1" or mode == "Fixed Consistency":
        b = torch.ones_like(q)
    elif rule == "linear_q":
        b = q
    elif rule == "bell_4q1mq":
        b = 4 * q * (1 - q)
    else:
        raise ValueError(f"Unknown consistency weight {rule}")
    a_ce = a * ce
    if mode in ("Random Downweight", "Loss Downweight"):
        return a_ce.mean()
    shifted = model(normalize(weak_view(x, seed)))
    return (a_ce + local["lambda"] * b * js(logits.softmax(1), shifted.softmax(1))).mean()


def train_local(model, opt, dataset, cfg, device, seed, steps, mode="CE", scores=None, scale=None):
    if dataset.role != "private_train":
        raise ValueError("Local updates require private_train")
    model.train()
    for n, positions in enumerate(sample_batches(dataset, cfg["batch_size"], steps, seed)):
        x, y, ids = batch_at(dataset, positions, device)
        if mode == "CE":
            q = None
        elif mode == "Fixed Consistency":
            q = torch.ones(len(y), device=device)
        else:
            q = scores.weights(ids, mode, scale, device)
        loss = local_loss(model, x, y, mode, cfg["local"], seed+n, q)
        step(loss, model, opt, cfg["clip"])
    return steps
