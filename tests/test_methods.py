from copy import deepcopy

import numpy as np
import torch

from core.runtime import step
from methods.work1_local_value.local import local_loss
from methods.work2_freq_kd.frequency import frequency_views
from methods.work2_freq_kd.kd import relay_weights


def test_local_weight_applied_once_and_detached():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    x = torch.rand(2, 3, 2, 2)
    y = torch.tensor([0, 1])
    q = torch.full((2,), .5, requires_grad=True)
    opts = {"a_min": .2, "consistency_weight": "const_1", "lambda": 0}
    base = local_loss(model, x, y, "CE", opts, 1)
    changed = local_loss(model, x, y, "Candidate Quality-Modulated", opts, 1, q)
    torch.testing.assert_close(changed, .6*base)
    changed.backward()
    assert q.grad is None


def test_deepcopy_preserves_optimizer_binding_and_momentum():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    x = torch.ones(4, 2)
    step(model(x).sum(), model, opt, None)
    saved = model.weight.detach().clone()
    branch, branch_opt = deepcopy((model, opt))
    torch.testing.assert_close(branch_opt.state[branch.weight]["momentum_buffer"],
                               opt.state[model.weight]["momentum_buffer"])
    step(branch(x).sum(), branch, branch_opt, None)
    torch.testing.assert_close(model.weight, saved)
    assert not torch.equal(branch.weight.detach(), saved)


def test_fft_and_full_frequency_fallback():
    x = torch.rand(4, 3, 32, 32)
    for eps in (0, .3, .6):
        views = frequency_views(x, eps, .17, .03)
        assert views["full"] is x
        for result in views.values():
            assert result.isfinite().all() and result.amin() >= 0 and result.amax() <= 1
            if not eps:
                torch.testing.assert_close(x, result)
    probs = [{b: torch.rand(4, 10).softmax(1) for b in views} for _ in range(3)]
    cfg = {"frequency": {"gamma_full": .6, "score_temperature": .1}}
    peers, weights = relay_weights(probs, 0, "frequency_routed", cfg, np.random.default_rng(2))
    assert peers == [1, 2]
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert weights[:, 0].sum() >= .6
