# Is EUR/USD normally distributed, and does a normal VaR underestimate risk?
# Usage: python var_eurusd.py   (downloads ECB data the first time)
import os
import csv
import urllib.request
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata"
DATA_FILE = "data/eurusd.csv"
WINDOW = 500       # number of past returns used to estimate VaR
N_DRAWS = 10000    # number of Monte Carlo draws per day
LEVELS = [0.95, 0.99]
METHODS = ["Normal", "Historical", "Monte Carlo"]


def load_rates():
    # Download the file once and save it, so we do not download it every run
    if not os.path.exists(DATA_FILE):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, DATA_FILE)

    dates = []
    rates = []
    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["OBS_VALUE"] == "":
                continue  # some days have no value
            dates.append(row["TIME_PERIOD"])
            rates.append(float(row["OBS_VALUE"]))

    # Sort so the oldest day comes first (dates like 2020-05-17 sort correctly as text)
    order = np.argsort(dates)
    dates = np.array(dates)[order].astype("datetime64[D]")
    rates = np.array(rates)[order]
    return dates, rates


def log_returns(rates):
    # r_t = 100 * ln(P_t / P_(t-1)), in percent
    return 100 * np.diff(np.log(rates))


def var_normal(window, level):
    # Assume the returns are normal: the cutoff is mean + z * std
    z = stats.norm.ppf(1 - level)  # negative number, for example -2.33 at 99%
    return -(np.mean(window) + z * np.std(window))  # minus sign: VaR is a positive loss


def var_historical(window, level):
    # No distribution assumed: take the 1% (or 5%) worst return in the window
    return -np.quantile(window, 1 - level)


def var_monte_carlo(window, level):
    # Simulate returns from a normal with the same mean and std as the window
    draws = np.random.normal(np.mean(window), np.std(window), N_DRAWS)
    return -np.quantile(draws, 1 - level)


def jarque_bera(returns):
    # Test 1: Jarque-Bera. H0: the returns are normal (skewness 0, excess kurtosis 0)
    n = len(returns)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # excess kurtosis, 0 for a normal distribution
    jb = n / 6 * (s**2 + k**2 / 4)
    p_value = stats.chi2.sf(jb, 2)  # chi-square with 2 degrees of freedom
    return jb, p_value


def kupiec_test(n_days, n_exceed, p):
    # Test 2: Kupiec. H0: the share of exceedance days really is p (1% or 5%)
    rate = n_exceed / n_days
    loglik_h0 = (n_days - n_exceed) * np.log(1 - p) + n_exceed * np.log(p)
    # log(0) is not allowed, so a term with 0 exceedances (or all days) is left out
    loglik_data = 0
    if n_exceed > 0:
        loglik_data = loglik_data + n_exceed * np.log(rate)
    if n_exceed < n_days:
        loglik_data = loglik_data + (n_days - n_exceed) * np.log(1 - rate)
    lr = -2 * (loglik_h0 - loglik_data)
    p_value = stats.chi2.sf(lr, 1)  # chi-square with 1 degree of freedom
    return lr, p_value


def backtest(returns):
    # For every day, estimate VaR from the WINDOW days BEFORE it and save it
    np.random.seed(42)  # same random numbers every run
    var_lists = {}
    for method in METHODS:
        for level in LEVELS:
            var_lists[(method, level)] = []

    for day in range(WINDOW, len(returns)):
        window = returns[day - WINDOW:day]  # ends before "day", so no look-ahead
        for level in LEVELS:
            var_lists[("Normal", level)].append(var_normal(window, level))
            var_lists[("Historical", level)].append(var_historical(window, level))
            var_lists[("Monte Carlo", level)].append(var_monte_carlo(window, level))

    for key in var_lists:
        var_lists[key] = np.array(var_lists[key])  # arrays so we can compare with returns
    return var_lists


def make_figures(returns, test_returns, test_dates, var_lists):
    os.makedirs("figures", exist_ok=True)

    # Figure 1: histogram with a normal curve; log scale on y shows the tails
    x = np.linspace(returns.min(), returns.max(), 500)
    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=100, density=True, color="lightgray", label="Daily returns")
    plt.plot(x, stats.norm.pdf(x, np.mean(returns), np.std(returns)), label="Normal")
    plt.yscale("log")
    plt.ylim(bottom=1e-4)
    plt.xlabel("Daily log return (%)")
    plt.ylabel("Density (log scale)")
    plt.title("EUR/USD returns vs normal distribution")
    plt.legend()
    plt.savefig("figures/histogram.png", dpi=150)
    plt.close()

    # Figure 2: QQ plot. Points on the line = normal. Ends bending away = fat tails
    plt.figure(figsize=(6, 6))
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title("QQ plot of returns against the normal")
    plt.savefig("figures/qq_plot.png", dpi=150)
    plt.close()

    # Figure 3: returns and the 99% VaR as a negative number.
    # A return below the line is an exceedance
    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, test_returns, color="lightgray", linewidth=0.7, label="Daily return")
    plt.plot(test_dates, -var_lists[("Normal", 0.99)], label="-99% VaR normal")
    plt.plot(test_dates, -var_lists[("Historical", 0.99)], label="-99% VaR historical")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.title("99% VaR over the backtest period")
    plt.legend()
    plt.savefig("figures/var_lines.png", dpi=150)
    plt.close()


def scenario_to_2040(returns, last_date, last_rate):
    # Scenario analysis, NOT a forecast: simulate many possible futures for the exchange rate
    np.random.seed(42)
    end_date = np.datetime64("2040-12-31")
    first_new_day = last_date + np.timedelta64(1, "D")  # the day after the last data point
    n_days = int(np.busday_count(first_new_day, end_date))  # weekdays left
    n_paths = 5000

    # Bootstrap: each future day is a random draw from the real historical returns.
    # This keeps the fat tails that the normal distribution misses
    draws = np.random.choice(returns, size=(n_paths, n_days))
    # Add up the log returns day by day, then turn them back into exchange rates
    paths = last_rate * np.exp(np.cumsum(draws, axis=1) / 100)

    # At every future day: the 5th, 50th and 95th percentile over all the paths
    low = np.percentile(paths, 5, axis=0)
    middle = np.percentile(paths, 50, axis=0)
    high = np.percentile(paths, 95, axis=0)

    # x axis in years, np.busday_count counts 261 weekdays per year
    start_year = 2026 + (last_date - np.datetime64("2026-01-01")).astype(int) / 365
    years = start_year + np.arange(1, n_days + 1) / 261

    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.plot(years, paths[i], color="lightgray", linewidth=0.5)  # a few example paths
    plt.fill_between(years, low, high, alpha=0.3, label="5%-95% of scenarios")
    plt.plot(years, middle, label="Median scenario")
    plt.xlabel("Year")
    plt.ylabel("EUR/USD")
    plt.title("Monte Carlo scenarios for EUR/USD to 2040 (not a forecast)")
    plt.legend()
    plt.savefig("figures/fan_2040.png", dpi=150)
    plt.close()

    print()
    print("Scenario analysis to 2040,", n_paths, "paths, start rate", last_rate)
    print("Rate at the end - 5%:", round(low[-1], 3), " median:", round(middle[-1], 3), " 95%:", round(high[-1], 3))


def main():
    dates, rates = load_rates()
    returns = log_returns(rates)
    print("Data:", dates[0], "to", dates[-1], "-", len(returns), "returns")

    # Part 1: are the returns normal?
    print("Mean:", round(np.mean(returns), 4), " std:", round(np.std(returns), 4))
    print("Skewness:", round(stats.skew(returns), 4),
          " excess kurtosis:", round(stats.kurtosis(returns), 4), "(both 0 for a normal)")
    jb, jb_p = jarque_bera(returns)
    print("Jarque-Bera:", round(jb, 1), " p-value:", jb_p)

    # Part 2: backtest the three VaR methods
    var_lists = backtest(returns)
    test_returns = returns[WINDOW:]
    test_dates = dates[1:][WINDOW:]  # a return belongs to the later of its two dates
    days = len(test_returns)

    print()
    print("Backtest:", days, "days, window", WINDOW, "days")
    print("Method        Level  AvgVaR%  Exceed  Expected  Kupiec p-value")
    for level in LEVELS:
        for method in METHODS:
            var_values = var_lists[(method, level)]
            # an exceedance is a day when the loss is bigger than the VaR
            n_exceed = int(np.sum(test_returns < -var_values))
            expected = (1 - level) * days
            lr, p_value = kupiec_test(days, n_exceed, 1 - level)
            print(f"{method:<13} {level:<5} {np.mean(var_values):8.3f} {n_exceed:7d} {expected:9.1f} {p_value:15.4f}")

    make_figures(returns, test_returns, test_dates, var_lists)
    scenario_to_2040(returns, dates[-1], rates[-1])
    print("Figures saved in figures/")


if __name__ == "__main__":
    main()
