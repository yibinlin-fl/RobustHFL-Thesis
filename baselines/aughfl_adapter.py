"""AugHFL core on frozen splits: AugMix/JSD and inverse-divergence teacher weights."""
import torch
from torch.nn import functional as F

from baselines.augmix import augmix_views, local_augmix
from baselines.common import train_rounds
from baselines.fedmd_kd import kd_loss, other_clients, proxy_batches
from baselines.local_ce import local_seed
from core.data import setup_data
from core.logger import tensor_bytes
from core.runtime import cli, initialize, normalize, read_args, step


@torch.no_grad()
def public_quality(model, views, temperature):
    model.eval()
    original, a, b = model(normalize(torch.cat(views))).chunk(3)
    p = original.softmax(1)
    div = sum(F.kl_div(z.log_softmax(1), p, reduction="batchmean") for z in (a, b))
    return (original/temperature).softmax(1), div.clamp_min(1e-7).reciprocal()


def aughfl_update(cfg, data, device):
    def update(clients, round_no):
        temp = cfg["temperature"]
        steps = comm = 0
        for n, x in enumerate(proxy_batches(cfg, data, device, round_no)):
            views = augmix_views(x, cfg["seed"]+10_000*round_no+n)
            probabilities, scores = zip(*(public_quality(model, views, temp) for model, _ in clients))
            comm += tensor_bytes(*probabilities, *scores)
            for s, (model, opt) in enumerate(clients):
                peers = other_clients(len(clients), s)
                weights = torch.stack([scores[k] for k in peers])
                weights = weights / weights.sum()
                target = (weights[:, None, None]*torch.stack([probabilities[k] for k in peers])).sum(0)
                model.eval()
                step(kd_loss(model, x, target, temp), model, opt, cfg["clip"])
                steps += 1
                comm += tensor_bytes(target)
        return steps, comm
    return update


def run(cfg, data, device):
    return train_rounds(cfg, data, device, "aughfl_adapter", local_augmix(cfg, data, device, local_seed),
                        aughfl_update(cfg, data, device))


if __name__ == "__main__":
    cfg = read_args(cli(__doc__, "configs/base.yaml").parse_args())
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device)
