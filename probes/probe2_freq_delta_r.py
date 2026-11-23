"""Probe 2: fixed-initialization frequency interventions and true local-validation ΔR."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from baselines import fedmd_kd
from core.data import setup_data
from core.evaluator import local_accuracy, ranking, summarize_clients
from core.logger import Budget, RunLog, tensor_bytes
from core.runtime import batch_at, cli, file_hash, initialize, read_args, read_checkpoint, sample_batches, step
from core.splits import digest
from methods.work2_freq_kd.frequency import BANDS, candidate_scores, entropy, frequency_views
from methods.work2_freq_kd.kd import KD_MODES, frequency_update


def build_public_views(ds, cfg, device, seed):
    opts = cfg["probe2"]
    freq = cfg["frequency"]
    positions = list(sample_batches(ds, cfg["batch_size"], opts["delta_steps"], seed))
    batches = [frequency_views(batch_at(ds, batch, device), freq["eps"], freq["cutoff"], freq["transition"])
               for batch in positions]
    return batches, digest([[ds.ids[i] for i in batch] for batch in positions])


def grid(path, views, label):
    count = min(6, len(views["full"]))
    canvas = Image.new("RGB", (3*104, 26+count*104), "white")
    draw = ImageDraw.Draw(canvas)
    for col, band in enumerate(BANDS):
        draw.text((col*104+4, 5), band, fill="black")
        for row, tensor in enumerate(views[band][:count]):
            rgb = (tensor.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
            tile = Image.fromarray(rgb).resize((96, 96), Image.Resampling.NEAREST)
            canvas.paste(tile, (104*col+4, 26+104*row))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, pnginfo=None)
    print(f"{label} FFT grid: {path}", flush=True)


def pixel_stats(log, views, predictions, domain, start_hash):
    for band in BANDS:
        x = views[band]
        shifted = (x - views["full"]).abs().mean().item()
        displacement = torch.stack([
            (p[band]-p["full"]).abs().sum(-1).mean()/2 for p in predictions
        ]).mean().item()
        values = {"minimum": x.amin().item(), "maximum": x.amax().item(),
                  "mean": x.mean().item(), "std": x.std().item(), "pixel_mean_abs_shift": shifted,
                  "prediction_total_variation": displacement}
        log.row("frequency_sanity", slice_name=domain, ckpt_hash=start_hash, band=band, **values)
        print(f"{domain} {band} {values}", flush=True)


def intervention(origin, batches, targets, band, cfg, device, data, student, robust_eval=None):
    model, opt = deepcopy(origin)
    model.eval()
    for views, target in zip(batches, targets):
        loss = fedmd_kd.kd_loss(model, views[band], target, cfg["temperature"])
        step(loss, model, opt, cfg["clip"])
    if robust_eval is not None:
        return robust_eval(model, student), local_accuracy(model, data.val_clean[student], cfg, device)
    return local_accuracy(model, data.val_robust[student], cfg, device)


def run(cfg, data, device, checkpoint_dir, *, robust_eval=None):
    log = RunLog(cfg, data, "probe2")
    checkpoint_dir = Path(checkpoint_dir)
    clients, hashes = [], []
    for i, client in enumerate(cfg["clients"]):
        path = checkpoint_dir / f"client_{i}.pt"
        clients.append(read_checkpoint(path, cfg, device, i, data.split_hash, data.view_hash))
        hashes.append(file_hash(path))
    start_hash = digest(hashes)
    log.json_event({"event": "source_checkpoints", "directory": str(checkpoint_dir),
                    "ckpt_hashes": hashes, "ckpt_hash": start_hash, **log.hashes})
    count = len(clients)
    opts = cfg["probe2"]
    if opts["delta_steps"] < 1 or opts["comparison_rounds"] < 1 or opts["repeats"] < 1 or cfg["kd_steps"] < 1:
        raise ValueError("Probe 2 requires positive update, comparison and repetition budgets")
    initial = [robust_eval(model, i) if robust_eval is not None
               else local_accuracy(model, data.val_robust[i], cfg, device) for i, (model, _) in enumerate(clients)]
    if robust_eval is not None:
        clean_initial = [local_accuracy(model, data.val_clean[i], cfg, device)
                         for i, (model, _) in enumerate(clients)]
    for domain_no, domain in enumerate(opts["domains"]):
        ds = data.proxies[domain]
        matrix = []
        for replicate in range(opts["repeats"]):
            public, batch_hash = build_public_views(ds, cfg, device, cfg["seed"]+80_000+replicate)
            predictions = [[{b: fedmd_kd.predict(model, views[b], 1) for b in BANDS}
                            for model, _ in clients] for views in public]
            if replicate == 0:
                pixel_stats(log, public[0], predictions[0], domain, start_hash)
                if domain_no == 0:
                    grid(Path(opts.get("sanity_image") or (log.path / "freq_views_sanity.png")),
                         public[0], domain)
                    if opts.get("save_view_pixels"):
                        np.savez_compressed(log.path / "frequency_view_pixels.npz",
                                            **{b: public[0][b][:6].cpu().numpy() for b in BANDS})
            targets = [[{b: fedmd_kd.predict(model, views[b], cfg["temperature"]) for b in BANDS}
                        for model, _ in clients] for views in public]
            for s, origin in enumerate(clients):
                for k in fedmd_kd.other_clients(count, s):
                    others = [i for i in range(count) if i not in (k, s)]
                    for band in BANDS:
                        scores = [candidate_scores(pred[k][band], [pred[k][b] for b in BANDS if b != band],
                                                   pred[s][band],
                                                   [pred[i][band] for i in others]) for pred in predictions]
                        means = {name: float(np.mean([row[name] for row in scores])) if scores[0][name] is not None
                                 else None for name in scores[0]}
                        teacher_targets = [target[k][band] for target in targets]
                        with Budget(device) as budget:
                            after = intervention(origin, public, teacher_targets, band, cfg, device, data, s, robust_eval)
                            budget.steps = len(public)
                            budget.comm_bytes = 2 * tensor_bytes(*teacher_targets)
                        extra = {}
                        if robust_eval is not None:
                            after, clean_after = after
                            probs = torch.cat([pred[k][band] for pred in predictions])
                            counts = probs.argmax(1).bincount(minlength=probs.shape[1])
                            extra = {"clean_before": clean_initial[s], "clean_after": clean_after,
                                     "delta_clean": clean_after-clean_initial[s],
                                     "teacher_marginal_entropy": entropy(probs.mean(0)).item(),
                                     "teacher_argmax_classes": int(counts.count_nonzero().item()),
                                     "teacher_majority_fraction": counts.max().item()/len(probs)}
                        result = {"student": s, "student_model": cfg["clients"][s]["model"], "teacher": k,
                                  "teacher_model": cfg["clients"][k]["model"], "band": band,
                                  "replicate": replicate, "public_batch_sha256": batch_hash,
                                  "student_ckpt_sha256": hashes[s], "teacher_ckpt_sha256": hashes[k],
                                  "lr": origin[1].param_groups[0]["lr"], "robust_before": initial[s],
                                  "robust_after": after, "delta_r": after-initial[s], **means, **extra}
                        matrix.append(result)
                        log.row("student_teacher_band_gain", slice_name=domain, budget=budget.record(),
                                ckpt_hash=start_hash, **result)
        for student in [*range(count), "all"]:
            subset = [r for r in matrix if student == "all" or r["student"] == student]
            for metric in ("mi", "confidence", "cross_view_consistency", "teacher_consensus", "discrepancy"):
                log.row("gain_ranking", slice_name=domain, ckpt_hash=start_hash, student=student, signal=metric,
                        **ranking([r[metric] for r in subset], [r["delta_r"] for r in subset]))
        if opts.get("run_comparisons", True):
            run_comparisons(cfg, data, device, domain, log, clients, start_hash)


def run_comparisons(cfg, data, device, domain, log, initial, start_hash):
    comparison_cfg = {**cfg, "proxy": domain}
    for mode in KD_MODES:
        clients = deepcopy(initial)
        update = frequency_update(comparison_cfg, data, device, mode)
        with Budget(device) as budget:
            for r in range(cfg["probe2"]["comparison_rounds"]):
                n, comm = update(clients, r)
                budget.steps += n
                budget.comm_bytes += comm
            records = [{"corruption": data.corruptions[i].kind,
                        "clean_acc": local_accuracy(model, data.val_clean[i], cfg, device),
                        "mixed_acc": local_accuracy(model, data.val_mixed[i], cfg, device),
                        "corrupt_acc": local_accuracy(model, data.val_robust[i], cfg, device)}
                       for i, (model, _) in enumerate(clients)]
        log.row("kd_comparisons", slice_name=domain, budget=budget.record(),
                ckpt_hash=start_hash, mode=mode, **summarize_clients(records))


if __name__ == "__main__":
    parser = cli(__doc__, "configs/probe2.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints", help="Directory containing protocol-matched client_*.pt")
    source.add_argument("--prepare-checkpoints", action="store_true", help="Train standard KD before the probe")
    args = parser.parse_args()
    cfg = read_args(args)
    device = initialize(cfg)
    data = setup_data(cfg)
    checkpoint_dir = args.checkpoints
    if args.prepare_checkpoints:
        checkpoint_dir = fedmd_kd.run({**cfg, "out": str(Path(cfg["out"]) / "bootstrap")}, data, device)["checkpoint_dir"]
    run(cfg, data, device, checkpoint_dir)
