"""Plain local CE plus uniform cross-client distillation on an unlabeled proxy."""
from pathlib import Path

import torch
from torch.nn import functional as F

from baselines.common import train_rounds
from baselines.local_ce import ce_update
from core.data import setup_data
from core.logger import tensor_bytes
from core.runtime import batch_at, cli, initialize, normalize, read_args, sample_batches, step


@torch.no_grad()
def predict(model, x, temperature):
    model.eval()
    return (model(normalize(x)) / temperature).softmax(1)


def kd_loss(model, x, target, temperature):
    logits = model(normalize(x))
    return temperature**2 * F.kl_div((logits / temperature).log_softmax(1), target, reduction="batchmean")


def other_clients(count, i):
    return [k for k in range(count) if k != i]


def peer_target(predictions, peers):
    return torch.stack([predictions[k] for k in peers]).mean(0)


def proxy_batches(cfg, data, device, round_no):
    ds = data.proxies[cfg["proxy"]]
    for positions in sample_batches(ds, cfg["batch_size"], cfg["kd_steps"], cfg["seed"]+40_000+round_no):
        yield batch_at(ds, positions, device)


def uniform_update(cfg, data, device, choose_peers=other_clients):
    def update(clients, round_no):
        temperature = cfg["temperature"]
        steps = comm = 0
        for x in proxy_batches(cfg, data, device, round_no):
            predictions = [predict(model, x, temperature) for model, _ in clients]
            comm += tensor_bytes(*predictions)
            for i, (model, opt) in enumerate(clients):
                peers = choose_peers(len(clients), i)
                if peers:
                    target = peer_target(predictions, peers)
                    model.eval()
                    step(kd_loss(model, x, target, temperature), model, opt, cfg["clip"])
                    steps += 1
                    comm += tensor_bytes(target)
        return steps, comm
    return update


def run(cfg, data, device):
    return train_rounds(cfg, data, device, "fedmd_kd", ce_update(cfg, data, device),
                        uniform_update(cfg, data, device))


if __name__ == "__main__":
    parser = cli(__doc__, "configs/base.yaml")
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    cfg = read_args(args)
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = str(Path(args.checkpoint_dir))
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device)
