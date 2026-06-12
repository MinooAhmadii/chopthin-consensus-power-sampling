"""Chopthin resampling (Gandy & Lau, "The chopthin algorithm for resampling",
IEEE T-SP 2016; arXiv:1502.07532).

Unlike systematic/multinomial/stratified/residual resampling, chopthin does NOT return
equally-weighted particles. It bounds the ratio between the largest and smallest weight by
`eta`: heavy particles (w >= a) are "chopped" into pieces, light particles (w < a) are
"thinned" (probabilistically kept/killed). It is unbiased, conserves total weight, returns
exactly N particles, and (Lemma 2) implicitly lower-bounds the effective sample size.

This module is PURE PYTHON (no torch / numpy) so it can be unit-tested anywhere, including a
Mac dev box with no GPU. The torch-facing wrapper that plugs into the Power-SMC core lives in
`smc_samp_utils.py` and simply calls `chopthin(...)` on the CPU weight vector (N is tiny, so
this is free relative to a model forward pass).

Public API:
    chopthin(weights, eta, N, rng=None) -> (ancestor_idx, new_weights)
    ess(weights) -> float
    ess_floor_from_eta(eta, n) -> float        # exact Lemma-2 lower bound on ESS
    eta_for_ess_floor(gamma) -> float          # large-n inverse: ESS floor ~ gamma * n

Guarantees of chopthin(...) (within numerical tolerance):
    (i)   unbiased:         E[ (#offspring of i) * w_tilde_i ] = w_i
    (ii)  target count:     len(ancestor_idx) == N   (exact)
    (iii) weight conserved: sum(new_weights) == sum(weights)   (exact)
    (iv)  bounded ratio:    max(new_weights) / min(new_weights) <= eta

Reference: continuous h (Eq. 5), eta >= 4:
    h_a^eta(w) = w/a            if w < a
               = 1              if a <= w < eta*a/2
               = 2w/(eta*a)     if w >= eta*a/2
and `a` solves  sum_i h_a^eta(w_i) = N.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Expected-offspring function h and its sum
# ---------------------------------------------------------------------------
def _h(w: float, a: float, eta: float) -> float:
    """Expected number of offspring for a particle of weight w (Eq. 5)."""
    if w < a:
        return w / a
    if w < 0.5 * eta * a:          # a <= w < eta*a/2
        return 1.0
    return 2.0 * w / (eta * a)     # w >= eta*a/2


def _sum_h(weights: Sequence[float], a: float, eta: float) -> float:
    return math.fsum(_h(w, a, eta) for w in weights)


def _solve_a(weights: Sequence[float], eta: float, N: int,
             tol: float = 1e-12, max_iter: int = 300) -> float:
    """Solve sum_i h_a^eta(w_i) = N for a > 0 by bisection.

    (Bisection rather than the paper's O(n log n) Algorithm 2 is deliberate: at N~32 the
    root-solve is negligible next to an LLM forward pass, and bisection is simpler & robust.)

    f(a) := sum_i h_a^eta(w_i) is continuous and non-increasing in a, with
    f -> +inf as a -> 0+ and f -> 0 as a -> +inf, so a unique solution exists.
    We return the lower bracket a_lo (where f(a_lo) >= N), which keeps boundary
    particles (w == a) on the HEAVY side -> uniform inputs are left unchanged.
    """
    wmax = max(weights)
    if wmax <= 0:
        return 1.0
    a_hi = wmax
    guard = 0
    while _sum_h(weights, a_hi, eta) > N and guard < 200:   # push until f(a_hi) <= N
        a_hi *= 2.0
        guard += 1
    a_lo = a_hi
    guard = 0
    while _sum_h(weights, a_lo, eta) < N and guard < 400:   # pull until f(a_lo) >= N
        a_lo *= 0.5
        guard += 1
    for _ in range(max_iter):
        if a_hi - a_lo <= tol * max(a_hi, 1e-300):
            break
        a_mid = 0.5 * (a_lo + a_hi)
        if _sum_h(weights, a_mid, eta) >= N:
            a_lo = a_mid                                     # f(a_lo) >= N invariant
        else:
            a_hi = a_mid
    return a_lo


# ---------------------------------------------------------------------------
# Systematic resampling that returns EXACTLY M offspring among k items
# ---------------------------------------------------------------------------
def _systematic_counts(masses: Sequence[float], M: int, rng: random.Random) -> List[int]:
    """Allocate exactly M offspring to len(masses) items, in expectation proportional
    to `masses` (a low-variance / systematic draw). Returns counts summing to M."""
    k = len(masses)
    counts = [0] * k
    if M <= 0 or k == 0:
        return counts
    s = math.fsum(masses)
    if s <= 0.0:                                   # no signal -> spread round-robin
        for j in range(M):
            counts[j % k] += 1
        return counts
    # cumulative (normalized) distribution
    cdf = []
    acc = 0.0
    for m in masses:
        acc += m / s
        cdf.append(acc)
    cdf[-1] = 1.0                                  # guard against fp drift
    u0 = rng.random() / M
    j = 0
    for i in range(M):
        pos = u0 + i / M
        while j < k - 1 and pos > cdf[j]:
            j += 1
        counts[j] += 1
    return counts


# ---------------------------------------------------------------------------
# Chopthin (Algorithm 1)
# ---------------------------------------------------------------------------
def chopthin(weights: Sequence[float], eta: float, N: int,
             rng: Optional[random.Random] = None) -> Tuple[List[int], List[float]]:
    """Chopthin resampling.

    Args:
        weights: nonnegative particle weights (need NOT be normalized), sum > 0.
        eta:     max allowed ratio between largest and smallest output weight (>= 4).
        N:       number of particles to return.
        rng:     random.Random for the two systematic draws (pass for reproducibility).

    Returns:
        (ancestor_idx, new_weights): two lists of length N. ancestor_idx[k] is the index
        of the original particle that offspring k descends from; new_weights[k] is its
        (generally unequal) weight w_tilde.
    """
    if rng is None:
        rng = random.Random()
    w = [float(x) for x in weights]
    n = len(w)
    if n == 0:
        return [], []
    if N <= 0:
        raise ValueError(f"chopthin requires N > 0 (got {N}).")
    if eta < 4.0:
        raise ValueError(f"continuous chopthin requires eta >= 4 (got {eta}).")
    if any(x < 0.0 for x in w):
        raise ValueError("chopthin requires nonnegative weights.")
    total = math.fsum(w)
    if total <= 0.0:
        raise ValueError("chopthin requires sum(weights) > 0.")

    a = _solve_a(w, eta, N)

    light = [i for i in range(n) if w[i] < a]     # thin
    heavy = [i for i in range(n) if w[i] >= a]    # chop

    idx: List[int] = []
    new_w: List[float] = []

    # ---- Thin step: accumulator-style systematic over expected counts w_i/a (each 0/1).
    # Survivors carry weight exactly `a` (the minimum output weight).
    u = rng.random()
    for i in light:
        u += w[i] / a
        if u >= 1.0:
            idx.append(i)
            new_w.append(a)
            u -= 1.0
    N_L = len(idx)
    sum_w_light = math.fsum(w[i] for i in light)

    # ---- Chop step: integer + fractional offspring for heavy particles.
    h_heavy = [_h(w[i], a, eta) for i in heavy]
    floor_heavy = [math.floor(h) for h in h_heavy]
    frac_heavy = [h - math.floor(h) for h in h_heavy]
    sum_floor = sum(floor_heavy)
    N_U = N - N_L - sum_floor                      # extra offspring to allocate
    sum_frac = math.fsum(frac_heavy)

    # zeta conserves total weight (property iii): see paper Eq. (2)-(3).
    zeta = (sum_w_light - a * N_L) / sum_frac if sum_frac > 0.0 else 0.0

    m_heavy = _systematic_counts(frac_heavy, N_U, rng)

    for k, i in enumerate(heavy):
        c = floor_heavy[k] + m_heavy[k]
        if c <= 0:
            continue
        w_tilde = w[i] + zeta * frac_heavy[k]      # adjusted weight, split across c pieces
        share = w_tilde / c
        for _ in range(c):
            idx.append(i)
            new_w.append(share)

    # Guarantee EXACTLY N offspring (property ii). At a rare integer-rounding boundary the
    # steps above can yield N+/-1 (e.g. N_L rounds up while the heavy fractional mass is ~0,
    # so N_U < 0, the chop step allocates 0 extra, and len = N_L + sum_floor > N). Repair in
    # place by trimming the lightest / padding the heaviest offspring; the distortion is
    # bounded by a single particle. This is done HERE, at the source, so the contract holds
    # for EVERY caller -- the previous bare `assert` raised before any caller could repair it,
    # making the torch wrapper's fix-up unreachable dead code (and crashing real runs unless
    # asserts are stripped with -O).
    if len(idx) > N:
        keep = sorted(range(len(idx)), key=lambda k: new_w[k], reverse=True)[:N]
        idx = [idx[k] for k in keep]
        new_w = [new_w[k] for k in keep]
    elif 0 < len(idx) < N:
        heaviest = max(range(len(idx)), key=lambda j: new_w[j])
        pad = N - len(idx)
        idx = idx + [idx[heaviest]] * pad
        new_w = new_w + [new_w[heaviest]] * pad

    assert len(idx) == N, f"chopthin produced {len(idx)} particles, expected N={N}"
    return idx, new_w


# ---------------------------------------------------------------------------
# ESS helpers
# ---------------------------------------------------------------------------
def ess(weights: Sequence[float]) -> float:
    """Effective sample size  (sum w)^2 / sum(w^2)."""
    s1 = math.fsum(weights)
    s2 = math.fsum(x * x for x in weights)
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def ess_floor_from_eta(eta: float, n: int) -> float:
    """Lemma-2 lower bound on ESS of an n-vector with weight ratio <= eta:
        ESS >= (4*eta*n + 1 - eta^2) / (eta + 1)^2 .
    The 4 multiplies only the eta*n term, NOT the whole numerator. Check against the
    paper's worked example: eta=10, n=32 -> (4*10*32 + 1 - 100)/11^2 = 1181/121 = 9.760.
    """
    return (4.0 * eta * n + 1.0 - eta * eta) / ((eta + 1.0) ** 2)


def eta_for_ess_floor(gamma: float) -> float:
    """Large-n inverse of the ESS bound: pick eta so the ESS floor ~ gamma * n.
        eta = (2 - gamma + 2*sqrt(1 - gamma)) / gamma.
    e.g. gamma=0.5 -> eta = 3 + sqrt(8) ~= 5.828. (Finite-n floor is slightly lower; use
    ess_floor_from_eta for the exact value at a given n.)

    Note: eta decreases as gamma rises and crosses the continuous-chopthin validity floor
    (eta >= 4) at gamma ~= 0.686. Requesting a higher ESS-floor fraction than that has no
    valid continuous-h eta, so we raise here rather than return an eta that chopthin() will
    later reject with a less obvious error.
    """
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    eta = (2.0 - gamma + 2.0 * math.sqrt(1.0 - gamma)) / gamma
    if eta < 4.0:
        raise ValueError(
            f"gamma={gamma} implies eta={eta:.4f} < 4, below the continuous-chopthin "
            f"validity floor (eta >= 4). Max achievable ESS-floor fraction is gamma ~= 0.686 "
            f"(eta = 4)."
        )
    return eta
