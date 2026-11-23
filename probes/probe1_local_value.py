"""Probe 1: measure diagnostic signals and perform paired group interventions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from core.data import setup_data
from core.evaluator import local_accuracy, ranking
from core.logger import Budget, RunLog
from core.runtime import batch_at, cli, initialize, init_client, normalize, read_args, save_checkpoint, step
from core.splits import digest
from methods.work1_local_value.local import LocalHistory, MODES, SIGNALS, observe, train_local


def paired_groups(dataset, values, ids, cfg, seed):
    rng = np.random.default_rng(seed)
    options = cfg["probe1"]
    valid = np.flatnonzero(np.isfinite(values))
    shuffled = rng.permutation(valid)
    ordered = shuffled[np.argsort(values[shuffled], kind="stable")]
    bins = np.array_split(ordered, options["quantile_bins"])
    if min(map(len, bins)) < options["group_size"]:
        return
    for idx, group in enumerate(bins):
        chosen = ids[rng.choice(group, options["group_size"], replace=False)].tolist()
        pool = set(ids[np.setdiff1d(valid, group)].tolist())
        if len(pool) < len(chosen):
            raise ValueError("Group too large for a disjoint replacement control")
        controls = []
        matched = 0
        for idx_to_replace in chosen:
            label = dataset.label_for(idx_to_replace)
            matches = sorted(pos for pos in pool if dataset.label_for(pos) == label)
            control = int(rng.choice(matches or sorted(pool)))
            matched += bool(matches)
            pool.remove(control)
            controls.append(control)
        yield idx, chosen, controls, matched / len(chosen)


def validate(model, data, i, cfg, device):
    return (local_accuracy(model, data.val_clean[i], cfg, device),
            local_accuracy(model, data.val_robust[i], cfg, device))


def intervention(origin, dataset, ids, cfg, device, scale):
    model, opt = deepcopy(origin)
    x, y, _ = batch_at(dataset, [dataset.position(idx) for idx in ids], device)
    model.train()
    for _ in range(cfg["probe1"]["intervention_steps"]):
        loss = F.cross_entropy(model(normalize(x)), y) * scale
        step(loss, model, opt, cfg["clip"])
    return model


def signal_stats(log, scores, dataset, index, spec, checksum):
    for scale, mapping in [("raw", scores.raw), ("percentile", scores.percentile)]:
        injected = np.isin(scores.ids, sorted(dataset.selected))
        groups = {"all": np.ones(len(scores.ids), dtype=bool), "clean": ~injected, "injected": injected}
        candidate = np.asarray([scores.quality[scale]["candidate"][i] for i in scores.ids])
        for membership, mask in groups.items():
            selected = candidate[mask]
            cutoff = 0.5
            accepted = int((selected >= cutoff).sum())
            log.row("quality_acceptance", slice_name=scale, ckpt_hash=checksum, client=index,
                    corruption=spec.kind, severity=spec.severity, ratio=spec.ratio, membership=membership,
                    cutoff=cutoff, n=len(selected), accepted=accepted,
                    acceptance_rate=accepted/len(selected) if len(selected) else None)
        for name, values in mapping.items():
            for membership, group in groups.items():
                mask = np.isfinite(values) & group
                selected_values = values[mask]
                quartiles = np.quantile(selected_values, [.25, .5, .75]).tolist() if mask.any() else [None]*3
                q_values = [scores.quality[scale]["candidate"][i] for i in scores.ids[mask]]
                log.row("signal_statistics", slice_name=scale, ckpt_hash=checksum, client=index, corruption=spec.kind,
                        severity=spec.severity, ratio=spec.ratio, membership=membership, signal=name,
                        n=int(mask.sum()), mean=float(selected_values.mean()) if mask.any() else None,
                        q25=quartiles[0], median=quartiles[1], q75=quartiles[2],
                        candidate_quality_mean=float(np.mean(q_values)) if q_values else None)


def run(cfg, data, device, *, client_indices=None):
    log = RunLog(cfg, data, "probe1")
    options = cfg["probe1"]
    if options["warmup_rounds"] < 1 or options["intervention_steps"] < 1 or options["comparison_rounds"] < 1:
        raise ValueError("Probe 1 requires warm-up, comparison updates and paired intervention updates")
    for i in range(len(data.train)) if client_indices is None else client_indices:
        dataset = data.train[i]
        spec = data.corruptions[i]
        model, opt = init_client(cfg["clients"][i]["model"], cfg, device, i)
        history = LocalHistory()
        seed = cfg["seed"] + 1000*i
        with Budget(device) as warmup:
            for r in range(options["warmup_rounds"]):
                observe(model, dataset, history, cfg, device, r, seed+r, score=False)
                warmup.steps += train_local(model, opt, dataset, cfg, device, seed+r, cfg["local_steps"])
            r = options["warmup_rounds"]
            scores = observe(model, dataset, history, cfg, device, r, seed+r)
            before_clean, before_robust = validate(model, data, i, cfg, device)
        checkpoint = Path(cfg["checkpoint_dir"]) / "probe1_warmup" / log.run_id / f"client_{i}.pt"
        checksum = save_checkpoint(checkpoint, model, opt, cfg, data, i)
        log.json_event({"event": "warmup_checkpoint", "client": i, "path": str(checkpoint),
                        "ckpt_hash": checksum, **log.hashes})
        log.row("warmup", slice_name="probe1", budget=warmup.record(), client=i,
                ckpt_hash=checksum, clean_acc=before_clean, corrupt_acc=before_robust)
        signal_stats(log, scores, dataset, i, spec, checksum)
        origin = (model, opt)
        for scale, mapping in [("raw", scores.raw), ("percentile", scores.percentile)]:
            for name, values in mapping.items():
                score_centers, effects = [], []
                for replicate in range(options.get("intervention_repeats", 1)):
                    group_seed = seed + 50 + replicate
                    for quantile, chosen, controls, matched in paired_groups(dataset, values, scores.ids, cfg, group_seed):
                        arms = {}
                        for arm, ids, weight in [("retain", chosen, 1.0), ("soft_downweight", chosen, options["soft_weight"]),
                                                 ("replace_control", controls, 1.0)]:
                            with Budget(device) as budget:
                                trained = intervention(origin, dataset, ids, cfg, device, weight)
                                clean, robust = validate(trained, data, i, cfg, device)
                                budget.steps = options["intervention_steps"]
                            arms[arm] = (clean, robust, budget.record(), ids, weight)
                        score_center = float(np.mean(values[[dataset.position(idx) for idx in chosen]]))
                        score_centers.append(score_center)
                        effects.append(arms["retain"][1] - arms["replace_control"][1])
                        for arm, (clean, robust, budget, ids, weight) in arms.items():
                            log.row("signal_marginal_intervention", slice_name=scale, budget=budget, client=i,
                                    ckpt_hash=checksum,
                                    model=cfg["clients"][i]["model"], corruption=spec.kind, severity=spec.severity,
                                    ratio=spec.ratio, signal=name, score_center=score_center, replicate=replicate,
                                    quantile=quantile, arm=arm, sample_count=len(ids), sample_ids_sha256=digest(ids),
                                    replacement_label_match_rate=matched, weight=weight, clean_before=before_clean,
                                    corrupt_before=before_robust, clean_acc=clean, corrupt_acc=robust,
                                    delta_clean=clean-before_clean, delta_robust=robust-before_robust,
                                    paired_delta_clean_vs_replacement=clean-arms["replace_control"][0],
                                    paired_delta_robust_vs_replacement=robust-arms["replace_control"][1])
                log.row("signal_ranking", slice_name=scale, ckpt_hash=checksum, client=i,
                        signal=name, **ranking(score_centers, effects))

            for mode in MODES:
                branch, branch_opt = deepcopy(origin)
                branch_history = deepcopy(history)
                branch_scores = scores
                with Budget(device) as budget:
                    for j in range(options["comparison_rounds"]):
                        if j and mode not in MODES[:2]:
                            branch_scores = observe(branch, dataset, branch_history, cfg, device,
                                                    r+j, seed+r+j)
                        budget.steps += train_local(branch, branch_opt, dataset, cfg, device, seed+100+j,
                                                    cfg["local_steps"], mode, branch_scores, scale)
                    clean, robust = validate(branch, data, i, cfg, device)
                log.row("local_comparisons", slice_name=scale, budget=budget.record(), client=i,
                        ckpt_hash=checksum,
                        model=cfg["clients"][i]["model"], corruption=spec.kind, severity=spec.severity,
                        ratio=spec.ratio, mode=mode, clean_acc=clean, corrupt_acc=robust,
                        delta_clean=clean-before_clean, delta_robust=robust-before_robust)
    print(f"Probe 1 tables: {log.path}", flush=True)


if __name__ == "__main__":
    cfg = read_args(cli(__doc__, "configs/probe1.yaml").parse_args())
    device = initialize(cfg)
    run(cfg, setup_data(cfg), device)
