"""Metrics for independent validation and a separately invoked final test."""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from core.runtime import STEP_DEADLINE, loader, normalize


@torch.inference_mode()
def accuracy(model, batches, device):
    model.eval()
    hits, count = 0, 0
    for x, y, *_ in batches:
        guard = STEP_DEADLINE.get()
        if guard is not None:
            guard()
        logits = model(normalize(x.to(device)))
        if guard is not None and not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite validation logits")
        hits += (logits.argmax(1) == y.to(device)).sum().item()
        count += len(y)
    if not count:
        raise ValueError("Empty evaluation split")
    return hits / count


def local_accuracy(model, dataset, cfg, device):
    if dataset.role != "client_local_val":
        raise ValueError("Local validation requires the client_local_val split")
    return accuracy(model, loader(dataset, cfg["eval_batch_size"]), device)


def evaluate_clients(models, data, cfg, device):
    records = []
    for idx, model in enumerate(models):
        records.append({
            "client": idx, "model": cfg["clients"][idx]["model"],
            "corruption": data.corruptions[idx].kind, "severity": data.corruptions[idx].severity,
            "ratio": data.corruptions[idx].ratio,
            "clean_acc": local_accuracy(model, data.val_clean[idx], cfg, device),
            "mixed_acc": local_accuracy(model, data.val_mixed[idx], cfg, device),
            "corrupt_acc": local_accuracy(model, data.val_robust[idx], cfg, device),
        })
    return records, summarize_clients(records)


def summarize_clients(records):
    by_type = {}
    for row in records:
        by_type.setdefault(row["corruption"], []).append(row["corrupt_acc"])
    by_type = {name: sum(values)/len(values) for name, values in sorted(by_type.items())}
    return {
        "clean_acc": float(np.mean([r["clean_acc"] for r in records])),
        "mixed_acc": float(np.mean([r["mixed_acc"] for r in records])),
        "corrupt_acc": float(np.mean([r["corrupt_acc"] for r in records])),
        "worst_client_acc": min(r["corrupt_acc"] for r in records),
        "corrupt_acc_by_type": json.dumps(by_type, sort_keys=True),
    }


def ranking(scores, gains):
    scores, gains = np.asarray(scores, dtype=float), np.asarray(gains, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(gains)
    scores, gains = scores[valid], gains[valid]
    positive = gains > 0
    corr = None
    if len(np.unique(scores)) > 1 and len(np.unique(gains)) > 1:
        corr = float(spearmanr(scores, gains).statistic)
    both = bool(positive.any() and not positive.all())
    top = scores == np.max(scores) if len(scores) else np.array([], dtype=bool)
    return {
        "n": len(scores), "n_positive": int(positive.sum()), "spearman": corr,
        "auroc": float(roc_auc_score(positive, scores)) if both else None,
        "auprc": float(average_precision_score(positive, scores)) if both else None,
        "positive_identification_rate": float(positive[top].mean()) if top.any() else None,
    }
