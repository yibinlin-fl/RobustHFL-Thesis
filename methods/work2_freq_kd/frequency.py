"""Light radial frequency interventions and proxy-only scoring."""
from math import log

import torch

from methods.work1_local_value.local import js


BANDS = ("full", "low", "high")


def frequency_views(x, eps, cutoff, transition):
    if not 0 <= eps <= 1 or not 0 < cutoff < 0.71 or transition <= 0:
        raise ValueError("Invalid radial filter parameters")
    if (x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point() or
            not torch.isfinite(x).all() or x.amin() < 0 or x.amax() > 1):
        raise ValueError("FFT inputs must be finite RGB NCHW in [0, 1]")
    h, w = x.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)[:, None]
    fx = torch.fft.rfftfreq(w, device=x.device, dtype=x.dtype)[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    low = torch.sigmoid((cutoff - radius) / transition)
    spectrum = torch.fft.rfft2(x)
    views = {"full": x}
    for name, mask in (("low", low), ("high", 1-low)):
        filtered = torch.fft.irfft2(spectrum * mask, s=(h, w))
        view = ((1-eps) * x + eps * filtered).clamp(0, 1)
        if not torch.isfinite(view).all() or view.amin() < 0 or view.amax() > 1:
            raise ValueError(f"Invalid FFT reconstruction for {name}")
        views[name] = view
    return views


def entropy(p):
    return -(p * p.clamp_min(1e-8).log()).sum(-1)


def mi(p):
    return (entropy(p.mean(0)) - entropy(p).mean()).clamp_min(0)


def candidate_scores(p, cross_views, student, remaining_teachers):
    consensus = None
    if remaining_teachers:
        reference = torch.stack(remaining_teachers).mean(0)
        consensus = 1 - js(p, reference).mean().item() / log(2)
    view_shift = torch.stack([js(p, view).mean() for view in cross_views]).mean().item()
    return {
        "mi": mi(p).item(),
        "confidence": p.max(-1).values.mean().item(),
        "cross_view_consistency": 1 - view_shift / log(2),
        "teacher_consensus": consensus,
        "discrepancy": js(student, p).mean().item(),
    }
