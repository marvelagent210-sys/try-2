"""Growth-optimal (fractional Kelly) allocation with an opportunity-cost hurdle.

Maximise   mu'w - (1 / 2k) w'Sigma w - sum_i [ b (w_i - w0_i)^+ + s (w0_i - w_i)^+ ]
subject to 0 <= w_i <= upper_i  and  sum_i w_i <= budget.

The first two terms are the second-order approximation of expected log growth
(excess over cash) scaled for fractional Kelly k; k = 1 is full Kelly. The last
term charges b per unit bought and s per unit sold. Putting idle cash to work
pays only b (the trading cost), while selling pays s = trading cost + switch
hurdle. A holding is therefore only replaced when the marginal growth of the
new idea beats it by more than b + s: "switch if a better opportunity exists"
without churning on noise.

Solved with FISTA (accelerated proximal gradient). The proximal step for the
asymmetric L1-around-w0 term plus the box and budget constraints is computed
exactly: each coordinate is a clipped asymmetric soft-threshold, and the budget
multiplier is found from the breakpoints of a piecewise-linear function.
"""

from __future__ import annotations

import numpy as np


def _soft_box(v, w0, tb, ts, upper, lam) -> np.ndarray:
    x = v - lam
    s = np.where(x > w0 + tb, x - tb, np.where(x < w0 - ts, x + ts, w0))
    return np.clip(s, 0.0, upper)


def prox(v: np.ndarray, w0: np.ndarray, tb: np.ndarray, ts: np.ndarray, upper: np.ndarray,
         budget: float) -> np.ndarray:
    """argmin_w 1/2 ||w - v||^2 + sum tb_i (w_i - w0_i)^+ + ts_i (w0_i - w_i)^+
    s.t. 0 <= w <= upper, sum w <= budget."""
    w = _soft_box(v, w0, tb, ts, upper, 0.0)
    total = w.sum()
    if total <= budget + 1e-12:
        return w
    # sum_i w_i(lam) is continuous, non-increasing and piecewise linear in lam,
    # with kinks only at these points; find the segment where it crosses budget.
    bps = np.concatenate([v - w0 - tb, v - w0 + ts, v - tb - upper, v + ts - upper, v - tb, v + ts])
    bps = np.unique(bps[bps > 0])
    sums = _soft_box(v[None, :], w0[None, :], tb[None, :], ts[None, :], upper[None, :],
                     bps[:, None]).sum(axis=1)
    k = int(np.argmax(sums <= budget))
    lam_a, sum_a = (0.0, total) if k == 0 else (bps[k - 1], sums[k - 1])
    lam_b, sum_b = bps[k], sums[k]
    lam = lam_b if sum_a == sum_b else lam_a + (sum_a - budget) / (sum_a - sum_b) * (lam_b - lam_a)
    return _soft_box(v, w0, tb, ts, upper, lam)


def trade_cost(w, w0, buy_cost, sell_cost) -> np.ndarray:
    """Per-asset penalty for moving from w0 to w."""
    return buy_cost * np.maximum(w - w0, 0.0) + sell_cost * np.maximum(w0 - w, 0.0)


def growth_objective(w, mu, cov, kelly_fraction, w0, buy_cost, sell_cost=None) -> float:
    sell_cost = buy_cost if sell_cost is None else sell_cost
    return float(mu @ w - 0.5 / kelly_fraction * w @ cov @ w - trade_cost(w, w0, buy_cost, sell_cost).sum())


def expected_log_growth(w, mu, cov) -> float:
    """Full-Kelly approximation of annual log growth in excess of cash."""
    return float(mu @ w - 0.5 * w @ cov @ w)


def optimize(mu, cov, w0, upper, budget: float, kelly_fraction: float, buy_cost, sell_cost=None,
             max_iter: int = 3000, tol: float = 1e-10) -> np.ndarray:
    mu = np.asarray(mu, float)
    cov = np.asarray(cov, float)
    n = len(mu)
    if n == 0:
        return np.zeros(0)
    w0 = np.asarray(w0, float)
    upper = np.asarray(upper, float)
    sell_cost = buy_cost if sell_cost is None else sell_cost
    buy = np.broadcast_to(np.asarray(buy_cost, float), (n,))
    sell = np.broadcast_to(np.asarray(sell_cost, float), (n,))
    lipschitz = max(np.linalg.eigvalsh(cov).max() / kelly_fraction, 1e-8)
    step = 1.0 / lipschitz
    tb, ts = buy * step, sell * step

    w = prox(np.clip(w0, 0.0, upper), w0, np.zeros(n), np.zeros(n), upper, budget)
    y, theta = w.copy(), 1.0
    for _ in range(max_iter):
        grad = cov @ y / kelly_fraction - mu
        w_new = prox(y - step * grad, w0, tb, ts, upper, budget)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        if np.dot(y - w_new, w_new - w) > 0:  # adaptive restart keeps FISTA monotone
            theta = 1.0
            y = w_new
        else:
            theta_new = (1 + np.sqrt(1 + 4 * theta * theta)) / 2
            y = w_new + ((theta - 1) / theta_new) * (w_new - w)
            theta = theta_new
        w = w_new
    return w
