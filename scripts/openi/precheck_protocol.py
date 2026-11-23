"""Common controls and measured model-qualification rules for the V2 probes."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from baselines.local_ce import local_seed
from core.corruptions import Corruption
from core.evaluator import local_accuracy
from core.runtime import batch_at, loader, merge, normalize, sample_batches, step
from scripts.openi.state import read_json, write_json
from scripts.openi.train import run_rounds


GAUSSIAN, BLUR = "gaussian_noise", "defocus_blur"
SCENARIOS = ("clean", "gaussian", "blur", "heterogeneous", "uniform_mixture")
STRATEGIES = ("clean_ce", "mixed_ce", "high_loss", "random")
MODELS = ("ResNet10", "ResNet10", "MobileNetV2", "MobileNetV2")


def clients_for(profile, mix):
    moderate, severe = mix["moderate"], mix["severe"]
    fraction = mix["moderate_fraction"] + mix["severe_fraction"]
    crossed = (GAUSSIAN, BLUR, BLUR, GAUSSIAN)
    clients = []
    for i, arch in enumerate(MODELS):
        kind = {"gaussian": GAUSSIAN, "blur": BLUR}.get(profile, crossed[i])
        doses = [(kind, moderate, mix["moderate_fraction"]), (kind, severe, mix["severe_fraction"])]
        if profile in {"clean", "clean_ce"}:
            doses = []
        elif profile == "uniform_mixture":
            doses = [(k, s, f/2) for k in (GAUSSIAN, BLUR) for _, s, f in doses]
        clients.append({"model": arch, "corruption": {"kind": kind, "severity": severe,
                         "ratio": fraction if doses else 0.0},
                        "corruption_parts": [{"kind": k, "severity": s, "ratio": f} for k, s, f in doses]})
    return clients


def training_config(cfg, root, input_root, profile):
    protocol = cfg["protocol"]
    return merge(cfg, {**protocol, "clients": clients_for(profile, cfg["mixture"]),
                       "seed": cfg["seed"], "download": False, "data_root": str(input_root),
                       "manifest": str(root / "splits_manifest.json")})


class Holdout(TensorDataset):
    role = "client_local_val"


class ValidationPanel:
    """Counterfactual corruptions of the same held-out positions for every condition."""
    def __init__(self, data, cfg):
        self.datasets = []
        for source in data.val_clean:
            fields = {"clean_acc": source}
            for kind in (GAUSSIAN, BLUR):
                for level in ("moderate", "severe"):
                    spec = Corruption(kind, cfg["mixture"][level], 1.0)
                    x = np.stack([spec.apply(image.astype(np.float32)/255, role=source.role,
                                             seed=[source.seed, idx])
                                  for image, idx in zip(source.images, source.ids, strict=True)])
                    fields[f"{kind}_{level}_acc"] = Holdout(torch.from_numpy(x),
                                                            torch.tensor(source.labels.tolist()))
            self.datasets.append(fields)

    def robust(self, model, client, cfg, device):
        values = [local_accuracy(model, self.datasets[client][f"{kind}_severe_acc"], cfg, device)
                  for kind in (GAUSSIAN, BLUR)]
        return float(np.mean(values))

    def evaluate(self, models, data, cfg, device):
        rows = []
        for i, model in enumerate(models):
            metrics = {key: local_accuracy(model, ds, cfg, device) for key, ds in self.datasets[i].items()}
            robust = (metrics[f"{GAUSSIAN}_severe_acc"] + metrics[f"{BLUR}_severe_acc"])/2
            rows.append({"client": i, "model": cfg["clients"][i]["model"], **metrics, "corrupt_acc": robust})
        summary = {key: float(np.mean([r[key] for r in rows]))
                   for key in rows[0] if key not in {"client", "model"}}
        summary["worst_client_acc"] = min(r["corrupt_acc"] for r in rows)
        return rows, summary


def weighted_ce(cfg, data, device, strategy):
    def update(clients, round_no):
        loss_sum = count = downweighted = 0
        for i, (model, opt) in enumerate(clients):
            ds = data.train[i]
            if ds.role != "private_train":
                raise ValueError("CE requires private_train input")
            seed = local_seed(cfg, i, round_no)
            rng = np.random.default_rng(seed+83)
            model.train()
            for positions in sample_batches(ds, cfg["batch_size"], cfg["local_steps"], seed):
                x, y, _ = batch_at(ds, positions, device)
                losses = F.cross_entropy(model(normalize(x)), y, reduction="none")
                objective = losses.mean()
                if strategy in {"high_loss", "random"}:
                    n = max(1, round(cfg["downweight"]["fraction"] * len(y)))
                    selection = (losses.detach().argsort(descending=True, stable=True)[:n] if strategy == "high_loss"
                                 else torch.as_tensor(rng.choice(len(y), n, replace=False), device=device))
                    weights = torch.ones_like(losses)
                    weights[selection] = cfg["downweight"]["weight"]
                    objective = (weights * losses).mean()
                    downweighted += n
                step(objective, model, opt, cfg["clip"])
                loss_sum += losses.detach().mean().item()
                count += 1
        return count, {"local_ce_loss": loss_sum/count, "downweighted_visits": downweighted}
    return update


@torch.inference_mode()
def training_level(model, dataset, cfg, device):
    model.eval()
    hits = total = 0
    predictions = torch.zeros(10, dtype=torch.long)
    labels = torch.zeros_like(predictions)
    for x, y, *_ in loader(dataset, cfg["eval_batch_size"]):
        logits = model(normalize(x.to(device)))
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite diagnostic predictions")
        pred = logits.argmax(1).cpu()
        hits += pred.eq(y).sum().item()
        total += len(y)
        predictions += pred.bincount(minlength=10)
        labels += y.bincount(minlength=10)
    return {"clean_acc": hits/total, "majority_label_baseline": labels.max().item()/total,
            "predicted_classes": predictions.count_nonzero().item(),
            "majority_prediction_fraction": predictions.max().item()/total, "n": total}


def qualify(models, data, cfg, device):
    gate = cfg["quality"]
    metrics = [training_level(m, data.val_clean[i], cfg, device) for i, m in enumerate(models)]
    for entry in metrics:
        entry["threshold"] = max(gate["min_clean_accuracy"], entry["majority_label_baseline"]+gate["majority_margin"])
        entry["passed"] = (entry["clean_acc"] >= entry["threshold"] and
                           entry["predicted_classes"] >= gate["min_predicted_classes"] and
                           entry["majority_prediction_fraction"] <= gate["max_majority_prediction"])
    return metrics


def train_arm(cfg, data, panel, device, train_root, label, metadata, notify, *, method="local_ce",
              strategy="mixed_ce", retain=()):
    checkpoints = run_rounds(cfg, data, device, method, train_root, metadata["run_id"],
                             {**metadata["identity"], "arm": label}, lambda r, final: notify(label, r, final),
                             local_update=weighted_ce(cfg, data, device, strategy),
                             evaluator=panel.evaluate, retain_rounds=retain, compact=True)
    return checkpoints


def state_file(root, round_no):
    from core.runtime import file_hash
    folder = root / "rounds" / f"{round_no:06d}"
    path = folder / "state.pt"
    if file_hash(path) != read_json(folder / "receipt.json")["state_sha256"]:
        raise ValueError(f"Snapshot SHA-256 mismatch: {path}")
    return path
