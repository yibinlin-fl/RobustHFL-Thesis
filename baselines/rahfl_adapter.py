"""RAHFL core under freeze: AugMix, dual contrastive loss, asymmetric validation routing."""
import torch
from torch.nn import functional as F

from baselines.augmix import local_augmix
from baselines.common import train_rounds
from baselines.fedmd_kd import other_clients, uniform_update
from baselines.local_ce import local_seed
from core.data import setup_data
from core.evaluator import local_accuracy
from core.runtime import cli, initialize, normalize, read_args
from methods.work1_local_value.local import weak_view


def dual_contrastive(features, y, x, model, seed):
    original, strong, _ = features.chunk(3)
    _, weak = model(normalize(weak_view(x, seed)), return_feat=True)
    embeddings = torch.cat([original, weak])
    labels = y.repeat(2)
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=x.device)
    same = (labels[:, None] == labels[None, :]) & ~diagonal
    similarities = (embeddings @ embeddings.T / .2).masked_fill(diagonal, -torch.inf)
    log_probs = similarities.log_softmax(1).masked_fill(~same, 0)
    supcon = -(log_probs.sum(1)/same.sum(1)).mean() * (.2/.07)
    anchors = torch.cat([weak, original]).T
    target = ((weak @ anchors).exp()/.2).softmax(1).detach()
    distribution = ((strong @ anchors).exp()/.2).log_softmax(1)
    return supcon + F.kl_div(distribution, target, reduction="batchmean")


def rahfl_update(cfg, data, device):
    def update(clients, round_no):
        quality = [local_accuracy(model, data.val_robust[i], cfg, device)
                   for i, (model, _) in enumerate(clients)]

        def choose(count, student):
            return [i for i in other_clients(count, student) if quality[i] >= quality[student]]

        steps, comm = uniform_update(cfg, data, device, choose)(clients, round_no)
        return steps, comm + 4 * len(clients)
    return update


def run(cfg, data, device):
    return train_rounds(cfg, data, device, "rahfl_adapter",
                        local_augmix(cfg, data, device, local_seed, dual_contrastive),
                        rahfl_update(cfg, data, device))


if __name__ == "__main__":
    cfg = read_args(cli(__doc__, "configs/base.yaml").parse_args())
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device)
