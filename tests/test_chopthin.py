"""Property tests for the Chopthin resampler (Gandy & Lau, 2016) as used by CCPS."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chopthin as ct  # noqa: E402

ETA = 3 + math.sqrt(8)


def _random_weights(rng, n):
    kind = rng.choice(("uniform", "lognormal", "spiky", "tiny"))
    if kind == "uniform":
        return [rng.random() + 1e-3 for _ in range(n)]
    if kind == "lognormal":
        return [math.exp(rng.gauss(0, 2)) for _ in range(n)]
    if kind == "spiky":
        w = [1e-6] * n
        w[rng.randrange(n)] = 1.0
        return w
    return [rng.random() * 1e-9 for _ in range(n)]


def _check(w, eta, N, rng):
    idx, new_w = ct.chopthin(w, eta, N, rng)
    assert len(idx) == N and len(new_w) == N
    assert all(0 <= i < len(w) for i in idx)
    assert all(x > 0 for x in new_w)
    assert math.isclose(math.fsum(new_w), math.fsum(w), rel_tol=1e-9), "total weight not conserved"
    assert max(new_w) / min(new_w) <= eta * (1 + 1e-9), "weight ratio bound violated"
    return idx, new_w


def test_properties_random():
    rng = random.Random(0)
    for _ in range(3000):
        n = rng.choice((1, 2, 3, 8, 16, 32, 64))
        N = rng.choice((n, 8, 16, 32, 64))
        eta = rng.choice((4.0, ETA, 10.0, 100.0))
        _check(_random_weights(rng, n), eta, N, rng)


def test_paper_setting_ess_floor():
    """At eta = 3 + sqrt(8) the output ESS must stay near N/2 (paper Prop. 1, Lemma 2)."""
    rng = random.Random(1)
    floor = ct.ess_floor_from_eta(ETA, 32)
    worst = float("inf")
    for _ in range(2000):
        _, new_w = ct.chopthin(_random_weights(rng, 32), ETA, 32, rng)
        worst = min(worst, ct.ess(new_w))
    assert worst >= floor * (1 - 1e-9), f"ESS {worst:.3f} fell below the Lemma-2 floor {floor:.3f}"


def test_deterministic_given_rng():
    w = [math.exp(random.Random(5).gauss(0, 2)) for _ in range(32)]
    a = ct.chopthin(w, ETA, 32, random.Random(123))
    b = ct.chopthin(w, ETA, 32, random.Random(123))
    assert a == b


def test_unbiased_offspring_counts():
    """E[offspring weight of particle i] = w_i (property i), checked by Monte Carlo."""
    rng = random.Random(2)
    w = [1.0, 2.0, 3.0, 10.0, 0.05, 0.5, 4.0, 0.2]
    n = len(w)
    acc = [0.0] * n
    trials = 20000
    for _ in range(trials):
        idx, new_w = ct.chopthin(w, ETA, n, rng)
        for i, x in zip(idx, new_w):
            acc[i] += x
    for i in range(n):
        mean = acc[i] / trials
        assert abs(mean - w[i]) < 0.05 * max(w[i], 0.1), f"particle {i}: E[weight]={mean:.3f} vs {w[i]}"


def test_repair_conserves_weight():
    over = ([0, 1, 1, 2], [1.0, 2.0, 0.5, 3.0])
    idx, w = ct._force_exact_count(*over, N=3)
    assert len(idx) == 3 and math.isclose(math.fsum(w), 6.5)
    assert 0.5 not in w and 2.5 in w             # the dropped weight went to its sibling
    under = ([0, 1, 2], [1.0, 4.0, 2.0])
    idx, w = ct._force_exact_count(*under, N=4)
    assert len(idx) == 4 and math.isclose(math.fsum(w), 7.0)
    assert idx.count(1) == 2 and w[1] == 2.0 and w[3] == 2.0
    same = ct._force_exact_count([0, 1], [1.0, 1.0], N=2)
    assert same == ([0, 1], [1.0, 1.0])


def test_rejects_bad_inputs():
    import pytest
    with pytest.raises(ValueError):
        ct.chopthin([1.0, 2.0], 3.0, 2)         # eta < 4
    with pytest.raises(ValueError):
        ct.chopthin([1.0, -1.0], ETA, 2)
    with pytest.raises(ValueError):
        ct.chopthin([0.0, 0.0], ETA, 2)
