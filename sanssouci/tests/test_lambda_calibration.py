import numpy as np
from numpy.testing import assert_array_almost_equal
from sanssouci.lambda_calibration import get_perm_p, get_pivotal_stats
from sanssouci.lambda_calibration import get_permuted_p_values_one_sample
from sanssouci.row_welch import row_welch_tests
from sanssouci.reference_families import t_inv_linear
from scipy import stats


def test_get_perm_p():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    B = 100
    categ = rng.randint(2, size=n)
    pvals = get_perm_p(X, categ, B=B, row_test_fun=stats.ttest_ind)
    assert pvals.shape == (B, p)
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < B * p * .12

    stats_ = [stats.ks_2samp, stats.bartlett, stats.ranksums, stats.kruskal]
    for stat in stats_:
        pvals = get_perm_p(X, categ, B=B, row_test_fun=stat)
        assert pvals.min() > 1.e-7
        assert pvals.max() <= 1
        assert np.sum(pvals < .1) < B * p * .12

    pvals = get_perm_p(X, categ, B=B, row_test_fun=row_welch_tests)
    assert pvals.shape == (B, p)
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < B * p * .12


def test_get_permuted_p_values_one_sample():
    rng = np.random.RandomState(42)
    n = 20
    p = 50
    X = rng.randn(n, p)
    B = 100
    pvals = get_permuted_p_values_one_sample(X, B=B)
    assert pvals.shape == (B, p)
    assert (pvals == np.sort(pvals, axis=1)).all()
    assert pvals.min() > 1.e-7
    assert pvals.max() <= 1
    assert np.sum(pvals < .1) < B * p * .12


def test_get_pivotal_stat():
    B = 500
    p = 20
    rng = np.random.RandomState(42)
    p0 = np.sort(rng.rand(B, p), 1)
    t_inv = t_inv_linear

    piv_stat = get_pivotal_stats(p0, t_inv=t_inv, K=-1)
    assert piv_stat.shape == (B,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    assert_array_almost_equal(piv_stat, np.min(t_inv(p0)[:, :p], axis=1))

    piv_stat = get_pivotal_stats(p0, t_inv=t_inv_linear, K=1)
    assert piv_stat.shape == (B,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    assert_array_almost_equal(piv_stat, np.min(t_inv(p0)[:, :1], axis=1))

    piv_stat = get_pivotal_stats(p0, t_inv=t_inv_linear, K=p)
    assert piv_stat.shape == (B,)
    assert piv_stat.min() > 1.e-7
    assert piv_stat.max() < p0.max() * p
    assert isinstance(piv_stat, np.ndarray)
    assert_array_almost_equal(piv_stat, np.min(t_inv(p0)[:, :p], axis=1))

    piv_statB = get_pivotal_stats(p0, t_inv=t_inv_linear, K=p)
    piv_statAll = get_pivotal_stats(p0, t_inv=t_inv_linear, K=-1)
    assert_array_almost_equal(piv_statB, piv_statAll)
