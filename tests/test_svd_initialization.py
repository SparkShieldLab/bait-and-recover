import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from step4_multilayer import compute_svd_factors  # noqa: E402


def test_full_svd_is_exact_by_default() -> None:
    weight = torch.randn(12, 8)
    left, singular_values, right = compute_svd_factors(weight)

    reconstructed = left @ torch.diag(singular_values) @ right.T
    assert left.shape == (12, 8)
    assert singular_values.shape == (8,)
    assert right.shape == (8, 8)
    assert torch.allclose(reconstructed, weight, atol=1e-5, rtol=1e-5)


def test_lowrank_svd_is_reproducible_and_preserves_rng_state() -> None:
    weight = torch.randn(12, 8)
    rng_state = torch.random.get_rng_state()

    left1, singular_values1, right1 = compute_svd_factors(
        weight, method="lowrank", lowrank_q=4, lowrank_niter=3, seed=123
    )
    state_after_first = torch.random.get_rng_state()
    left2, singular_values2, right2 = compute_svd_factors(
        weight, method="lowrank", lowrank_q=4, lowrank_niter=3, seed=123
    )

    assert torch.equal(rng_state, state_after_first)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert left1.shape == (12, 4)
    assert singular_values1.shape == (4,)
    assert right1.shape == (8, 4)
    assert torch.allclose(singular_values1, singular_values2)
    assert torch.allclose(left1 @ left1.T, left2 @ left2.T)
    assert torch.allclose(right1 @ right1.T, right2 @ right2.T)
    assert torch.all(singular_values1[:-1] >= singular_values1[1:])
