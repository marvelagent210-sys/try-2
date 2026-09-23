import numpy as np
import pytest

from growthpm.optimizer import growth_objective, optimize, prox


def random_problem(n=6, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T / 20 + np.eye(n) * 0.02
    mu = rng.uniform(0.0, 0.05, n)
    return mu, cov


def test_matches_closed_form_kelly_when_unconstrained():
    mu, cov = random_problem()
    k = 0.5
    w = optimize(mu, cov, np.zeros(6), np.ones(6), 10.0, k, 0.0)
    expected = k * np.linalg.solve(cov, mu)
    assert np.all(expected > 0)  # interior solution, so the closed form applies
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_respects_box_and_budget_and_is_optimal():
    mu, cov = random_problem(seed=1)
    mu = mu * 5
    upper = np.full(6, 0.3)
    w = optimize(mu, cov, np.zeros(6), upper, 1.0, 0.5, 0.01)
    assert np.all(w >= -1e-12) and np.all(w <= upper + 1e-9)
    assert w.sum() <= 1.0 + 1e-9
    best = growth_objective(w, mu, cov, 0.5, np.zeros(6), 0.01)
    rng = np.random.default_rng(2)
    for _ in range(3000):
        z = np.clip(w + rng.normal(scale=0.02, size=6), 0, 0.3)
        if z.sum() > 1:
            z /= z.sum()
        assert growth_objective(z, mu, cov, 0.5, np.zeros(6), 0.01) <= best + 1e-9


def two_asset_switch(mu_new, buy_cost=0.004, sell_cost=0.064):
    """Hold 50% of A; B is a near-clone (so no diversification gain) but for its expected return."""
    cov = np.array([[0.04, 0.0396], [0.0396, 0.04]])
    mu = np.array([0.10, mu_new])
    w0 = np.array([0.5, 0.0])
    return optimize(mu, cov, w0, np.array([0.5, 0.5]), 0.5, 0.5, buy_cost, sell_cost)


def test_small_edge_does_not_trigger_a_switch():
    w = two_asset_switch(mu_new=0.15)   # edge 5% < buy + sell hurdle 6.8%
    np.testing.assert_allclose(w, [0.5, 0.0], atol=1e-9)


def test_large_edge_triggers_a_switch():
    w = two_asset_switch(mu_new=0.30)
    assert w[1] > 0.2 and w[0] < 0.3


def test_idle_cash_is_deployed_without_the_switch_hurdle():
    """A modest opportunity is bought with spare cash even though it could never
    justify selling something else to fund it."""
    cov = np.array([[0.04]])
    w = optimize(np.array([0.03]), cov, np.zeros(1), np.ones(1), 1.0, 0.5, 0.004, 0.064)
    assert w[0] == pytest.approx(0.5 * (0.03 - 0.004) / 0.04, rel=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_prox_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 4
    v = rng.normal(0.2, 0.3, n)
    w0 = rng.uniform(0, 0.4, n)
    tb = rng.uniform(0, 0.05, n)
    ts = rng.uniform(0, 0.10, n)
    upper = rng.uniform(0.1, 0.5, n)
    budget = 0.6
    w = prox(v, w0, tb, ts, upper, budget)
    assert w.sum() <= budget + 1e-9 and np.all(w >= 0) and np.all(w <= upper + 1e-12)

    def f(x):
        return 0.5 * np.sum((x - v) ** 2) + np.sum(tb * np.maximum(x - w0, 0) + ts * np.maximum(w0 - x, 0))

    for _ in range(4000):
        x = rng.uniform(0, upper)
        if x.sum() > budget:
            x *= budget / x.sum()
        assert f(x) >= f(w) - 1e-9
