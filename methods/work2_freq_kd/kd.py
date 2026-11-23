"""Matched-view KD comparisons and conservative student-conditioned frequency routing."""
import numpy as np
import torch

from baselines.fedmd_kd import kd_loss, other_clients, predict, proxy_batches
from core.logger import tensor_bytes
from core.runtime import step
from methods.work1_local_value.local import js
from methods.work2_freq_kd.frequency import BANDS, frequency_views, mi


KD_MODES = ("Full-Freq KD", "Fixed Low KD", "Fixed High KD", "Random Band KD",
            "Compute-Matched Multi-View KD", "MI-Weighted KD")


def loss_for_views(model, views, targets, weights, temp):
    model.eval()
    losses = torch.stack([kd_loss(model, views[b], targets[b], temp) for b in BANDS])
    return (losses * weights).sum()


def relay_weights(probabilities, s, mode, cfg, rng):
    peers = other_clients(len(probabilities), s)
    observations = torch.stack([torch.stack([mi(probabilities[k][b]) for b in BANDS]) for k in peers])
    if mode in KD_MODES[:4]:
        band = KD_MODES.index(mode)
        if band == 3:
            band = int(rng.integers(len(BANDS)))
        weights = torch.zeros_like(observations)
        weights[:, band] = 1 / len(peers)
    elif mode == "Compute-Matched Multi-View KD":
        weights = torch.full_like(observations, 1/observations.numel())
    elif mode in ("MI-Weighted KD", "frequency_routed"):
        if mode == "frequency_routed":
            gamma = cfg["frequency"]["gamma_full"]
            if not 0 <= gamma <= 1:
                raise ValueError("gamma_full must be in [0, 1]")
            discrepancy = torch.stack([torch.stack([js(probabilities[s][b], probabilities[k][b]).mean()
                                                    for b in BANDS]) for k in peers])
            observations = observations - discrepancy
        softness = cfg["frequency"]["score_temperature"]
        if softness <= 0:
            raise ValueError("score_temperature must be positive")
        weights = (observations / softness).flatten().softmax(0).reshape_as(observations)
        if mode == "frequency_routed":
            weights *= 1-gamma
            weights[:, 0] += gamma/len(peers)
    else:
        raise ValueError(f"Unknown frequency mode: {mode}")
    return peers, weights


def frequency_update(cfg, data, device, mode="frequency_routed"):
    def update(clients, round_no):
        temp = cfg["temperature"]
        freq = cfg["frequency"]
        rng = np.random.default_rng(cfg["seed"] + round_no)
        steps, comm = 0, 0
        for x in proxy_batches(cfg, data, device, round_no):
            views = frequency_views(x, freq["eps"], freq["cutoff"], freq["transition"])
            predictions = [{b: predict(model, views[b], temp) for b in BANDS} for model, _ in clients]
            comm += tensor_bytes(*(p for teacher in predictions for p in teacher.values()))
            for s, (model, opt) in enumerate(clients):
                peers, weights = relay_weights(predictions, s, mode, cfg, rng)
                band_weights = weights.sum(0)
                targets = {}
                for j, b in enumerate(BANDS):
                    probs = torch.stack([predictions[k][b] for k in peers])
                    if band_weights[j] > 0:
                        targets[b] = (probs * weights[:, j, None, None]).sum(0) / band_weights[j]
                    else:
                        targets[b] = probs.mean(0)
                comm += tensor_bytes(*targets.values(), band_weights)
                step(loss_for_views(model, views, targets, band_weights, temp), model, opt, cfg["clip"])
                steps += 1
        return steps, comm
    return update
