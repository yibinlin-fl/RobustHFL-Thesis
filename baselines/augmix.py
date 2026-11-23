"""Modern three-view AugMix and stable consistency losses shared by the adapters."""
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import AugMix

from core.runtime import batch_at, normalize, sample_batches, step


def augmix_views(x, seed):
    pipeline = AugMix(severity=3)
    uint8 = (x.cpu()*255).round().to(torch.uint8)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return x, *(torch.stack([pipeline(image) for image in uint8]).to(x)/255 for _ in range(2))


def three_view_js(probabilities):
    midpoint = torch.stack(probabilities).mean(0).clamp_min(1e-8).log()
    return sum(F.kl_div(midpoint, p, reduction="batchmean") for p in probabilities)/len(probabilities)


def local_augmix(cfg, data, device, local_seed, extra_loss=None):
    def update(clients, round_no):
        n = 0
        for i, (model, opt) in enumerate(clients):
            model.train()
            ds = data.train[i]
            seed = local_seed(cfg, i, round_no)
            for j, positions in enumerate(sample_batches(ds, cfg["batch_size"], cfg["local_steps"], seed)):
                x, y, _ = batch_at(ds, positions, device)
                views = augmix_views(x, seed+j)
                logits, features = model(normalize(torch.cat(views)), return_feat=True)
                first, second, third = logits.chunk(3)
                loss = F.cross_entropy(first, y) + 12*three_view_js([z.softmax(1) for z in (first, second, third)])
                if extra_loss:
                    loss = loss + extra_loss(features, y, x, model, seed+j)
                step(loss, model, opt, cfg["clip"])
                n += 1
        return n
    return update
