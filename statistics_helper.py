"""
    .mean_confidence_interval(data, confidence=0.95)
    .median_confidence_interval(data, confidence=.95)



"""

import pandas as pd
import numpy as np
import scipy.stats
import statistics
import math
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


class ECDFHelper:
    """
    An empirical distribution function provides a way to model and sample
    cumulative probabilities for a data sample that does not fit a standard
    probability distribution.

    As such, it is sometimes called the "empirical cumulative distribution function",
    or ECDF for short.

    https://machinelearningmastery.com/what-are-probability-distributions/
    https://machinelearningmastery.com/empirical-distribution-function-in-python/

    There are two main types of probability distribution functions we may need to sample; they are:
    . Probability Density Function (PDF).
    . Cumulative Distribution Function (CDF).

    The PDF returns the expected probability for observing a value.
        For discrete data, the PDF is referred to as a "Probability Mass Function" (PMF).

    The CDF returns the expected probability for observing a value less than or equal to a given value.

    Usage
    =====


    methods - public
    ----------------
    .fit()
    .plot_cdf()
    .get_cumulative_probability(x)
        .test_get_cumulative_probability(x)
    .with_ci_get_x(confidence_level: float = 0.95):
        . test_with_ci_get_x(cl=0.98):

    methods - internal
    ------------------

    """

    def __init__(self, df: pd.DataFrame):
        self._data = df
        self._fitted = False
        self._ecdf = None

    def fit(self):
        self._ecdf = ECDF(self._data)
        self._fitted = True

    def plot_cdf(self):
        plt.plot(self._ecdf.x, self._ecdf.y)
        plt.show()

    def get_cumulative_probability(self, x_val):
        return self._ecdf(x_val)

    def with_p_value_get_x(self, desired_p: float = 0.5):
        if not self._fitted:
            msg = "Please fit ECDF with data first, otherwise all values are None"
            print(f"\n{msg}")
            raise SystemError(msg)

        print("test ci get x v2")
        x_min = self._data.min()
        x_max = self._data.max()
        x_range = np.arange(x_min, x_max)
        x_range = np.append(x_range, x_max)

        p_values = []
        for idx, x in enumerate(x_range):
            p_val = self.get_cumulative_probability(x)
            p_values.append(p_val)

            if p_val >= desired_p:
                prev_p_val = p_values[idx-1]
                p_ratio = (desired_p - prev_p_val) / (p_val - prev_p_val)
                x_values_range = x_range[idx] - x_range[idx-1]
                final_x = x_range[idx - 1] + (p_ratio * x_values_range)
                return final_x, desired_p
        else:
            # if x does not exist in x_range, .get_cumulative_probability() returns p_value of None
            final_x = x_max
            final_p_value = 1.0
            return final_x, final_p_value


def test_get_cumulative_probability(x, show_chart=False):
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    zz = ECDFHelper(df['tht'])
    zz.fit()
    if show_chart:
        zz.plot_cdf()
    if isinstance(x, list):
        for i in x:
            print(f"P(x<{i}): {zz.get_cumulative_probability(i):,.3f}")
    else:
        print(f"P(x<{x}): {zz.get_cumulative_probability(x):,.3f}")


def test_with_p_value_get_x(p=0.5, show_chart=True):
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    # df.head()

    zz = ECDFHelper(df['tht'])
    zz.fit()
    if show_chart:
        zz.plot_cdf()
    # print(f"P(x<240): {zz.get_cumulative_probability(240):,.3f}")
    # print(f"P(x<661): {zz.get_cumulative_probability(661):,.3f}")
    p_value = p
    x_val, p = zz.with_p_value_get_x(desired_p=p_value)
    print(f"The probability of values being less than {x_val} is {p}.")


def mean_confidence_interval(data, confidence=0.95):
    # https://pythonguides.com/scipy-confidence-interval/
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    # m is mean
    return m, m-h, m+h


def median_confidence_interval(data, confidence=.95):
    dx = data.sort_values(ascending=True, ignore_index=True)
    factor = statistics.NormalDist().inv_cdf((1+confidence)/2)
    factor *= math.sqrt(len(dx))

    lower_interval = round(0.5*(len(dx)-factor))
    upper_interval = round(0.5*(1+len(dx)+factor))

    return dx[lower_interval], dx[upper_interval]


def test_median_confidence_interval():
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='Caffeine_batches', skiprows=6)

    cols = ['batch_num', 'caffeine_pct', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)

    data = df['caffeine_pct']
    cl = 0.95
    low, high = median_confidence_interval(data, cl)
    print(f"{cl*100}% Confidence Interval for Median is between {low:,.4f} and {high:,.4f}")


def test_mean_confidence_interval():
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='Caffeine_batches', skiprows=6)

    cols = ['batch_num', 'caffeine_pct', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)

    data = df['caffeine_pct']
    cl = 0.95
    m, m_l, m_h = mean_confidence_interval(data, confidence=cl)

    print(f"{cl*100}% Confidence Interval for Mean is between {m_l:,.4f} and {m_h:,.4f}")


if __name__ == "__main__":
    # test_mean_confidence_interval()
    # test_median_confidence_interval()
    X = [500, 501, 699, 700]
    test_get_cumulative_probability(X, show_chart=True)
    test_with_p_value_get_x(p=0.982, show_chart=False)
