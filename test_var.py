# Small tests of the helper functions. Run with: pytest
import numpy as np
from scipy import stats
import var_eurusd


def test_log_returns_known_values():
    # 100 * ln(110/100) and 100 * ln(99/110), computed by hand
    rates = np.array([100.0, 110.0, 99.0])
    returns = var_eurusd.log_returns(rates)
    assert len(returns) == 2
    assert abs(returns[0] - 100 * np.log(1.1)) < 1e-12
    assert abs(returns[1] - 100 * np.log(0.9)) < 1e-12


def test_normal_var_is_mean_plus_z_std():
    # Window with mean 0 and std 1 (half the days -1, half +1), so VaR = -(0 + z * 1) = 2.326 at 99%
    window = np.array([-1.0, 1.0] * 250)
    var_99 = var_eurusd.var_normal(window, 0.99)
    assert abs(var_99 - 2.3263478740) < 1e-6


def test_historical_var_on_known_array():
    # Returns -100, -99, ..., -1: the 5% quantile is -95.05, so VaR(95%) = 95.05
    window = np.arange(-100.0, 0.0)
    assert abs(var_eurusd.var_historical(window, 0.95) - 95.05) < 1e-9


def test_jarque_bera_matches_scipy():
    np.random.seed(1)
    returns = np.random.standard_t(4, size=2000)
    jb, p_value = var_eurusd.jarque_bera(returns)
    scipy_result = stats.jarque_bera(returns)
    assert abs(jb - scipy_result.statistic) < 1e-8
    assert p_value < 0.001  # t(4) data has fat tails, so normality is rejected


def test_kupiec_p_close_to_one_when_exceedances_as_expected():
    # 1% of 2500 days = 25 expected exceedances, and we observe exactly 25
    lr, p_value = var_eurusd.kupiec_test(2500, 25, 0.01)
    assert p_value > 0.99


def test_kupiec_small_p_when_far_off_and_zero_case_works():
    lr, p_value = var_eurusd.kupiec_test(2500, 60, 0.01)
    assert p_value < 0.001
    # zero exceedances must give a finite number, not nan (log(0) is avoided)
    lr_zero, p_zero = var_eurusd.kupiec_test(250, 0, 0.01)
    assert np.isfinite(lr_zero) and np.isfinite(p_zero)
