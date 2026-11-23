"""Independent A/B/C/D experiments with an explicit local-by-frequency interaction."""
from baselines.common import train_rounds
from baselines.fedmd_kd import uniform_update
from baselines.local_ce import ce_update, local_seed
from core.data import setup_data
from core.logger import RunLog
from core.runtime import cli, initialize, read_args
from core.splits import digest
from methods.work1_local_value.local import LocalHistory, observe, train_local
from methods.work2_freq_kd.kd import frequency_update


def value_update(cfg, data, device):
    histories = [LocalHistory() for _ in data.train]
    warmup = cfg["local"]["warmup_rounds"]

    def update(clients, round_no):
        steps = 0
        for i, (model, opt) in enumerate(clients):
            seed = local_seed(cfg, i, round_no)
            enabled = round_no >= warmup
            scores = observe(model, data.train[i], histories[i], cfg, device, round_no, seed, score=enabled)
            mode = "Candidate Quality-Modulated" if enabled else "CE"
            steps += train_local(model, opt, data.train[i], cfg, device, seed, cfg["local_steps"],
                                 mode, scores, cfg["local"]["scale"])
        return steps
    return update


def run(cfg, data, device, groups):
    if not groups or len(groups) != len(set(groups)) or set(groups) - set("ABCD"):
        raise ValueError("Select distinct groups from A/B/C/D")
    if cfg["kd_steps"] < 1:
        raise ValueError("A/B/C/D requires positive KD steps")
    log = RunLog(cfg, data, "abcd")
    if set(groups) & {"B", "D"} and not 1 <= cfg["local"]["warmup_rounds"] < cfg["rounds"]:
        raise ValueError("Local-value groups require at least one warm-up and one modulated round")
    results = {}
    for group in groups:
        local = value_update if group in "BD" else ce_update
        kd = frequency_update if group in "CD" else uniform_update
        results[group] = train_rounds(cfg, data, device, group, local(cfg, data, device),
                                      kd(cfg, data, device), log)
    if set(results) != set("ABCD"):
        return results
    budgets = {group: result["budget"] for group, result in results.items()}
    fixed_steps = {budget["steps"] for budget in budgets.values()}
    if len(fixed_steps) != 1:
        raise ValueError("A/B/C/D group step mismatch")
    checksum = digest({group: result["ckpt_hash"] for group, result in results.items()})
    for metric in ("clean_acc", "mixed_acc", "corrupt_acc", "worst_client_acc"):
        a, b, c, d = (results[g]["metrics"][metric] for g in "ABCD")
        log.row("interaction", slice_name="ABCD", ckpt_hash=checksum,
                metric=metric, A=a, B=b, C=c, D=d, interaction=(d-b)-(c-a), matched_steps=next(iter(fixed_steps)))
    return results


if __name__ == "__main__":
    parser = cli(__doc__, "configs/abcd.yaml")
    parser.add_argument("--groups", nargs="+", choices=list("ABCD"))
    args = parser.parse_args()
    cfg = read_args(args)
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device, args.groups or cfg["groups"])
