"""Purely local cross-entropy under the shared split protocol."""
from baselines.common import train_rounds
from core.data import setup_data
from core.runtime import cli, initialize, read_args
from methods.work1_local_value.local import train_local


def local_seed(cfg, client, round_no):
    return cfg["seed"] + 10_000 * (round_no+1) + client


def ce_update(cfg, data, device):
    def update(clients, round_no):
        return sum(train_local(model, opt, data.train[i], cfg, device, local_seed(cfg, i, round_no),
                               cfg["local_steps"]) for i, (model, opt) in enumerate(clients))
    return update


def run(cfg, data, device):
    return train_rounds(cfg, data, device, "local_ce", ce_update(cfg, data, device))


if __name__ == "__main__":
    cfg = read_args(cli(__doc__, "configs/base.yaml").parse_args())
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device)
