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

    """

    def __init__(self, df: pd.DataFrame):
        self._data = df
        self._ecdf = None

    def fit(self):
        self._ecdf = ECDF(self._data)

    def plot_cdf(self):
        plt.plot(self._ecdf.x, self._ecdf.y)
        plt.show()

    def get_cumulative_probability(self, x):
        return self._ecdf(x)

    def with_ci_get_x(self, confidence_level: float = 0.95):
        # todo: currently fn overshoots CL, run test_fn at cl=0.5 to see
        x_min = self._data.min()
        x_max = self._data.max()
        x_range = np.arange(x_min, x_max)

        final_x = None
        final_p_value = None

        for x in x_range:
            p_value = self._ecdf(x)
            if p_value >= confidence_level:
                # print(f"\nWe are {p_value * 100:,.2f}% confident that x will be less than {int(x)}. \n")
                final_x = x
                final_p_value = p_value
                break

        if final_p_value is None:
            final_x = x_max
            final_p_value = 1.0

        return final_x, final_p_value


def test_ecdf_helper():
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)

    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    df.head()

    zz = ECDFHelper(df['tht'])
    zz.fit()
    zz.plot_cdf()
    print(f"P(x<240): {zz.get_cumulative_probability(240):,.3f}")
    print(f"P(x<661): {zz.get_cumulative_probability(661):,.3f}")
    confidence_level = 0.98
    x, p_value = zz.with_ci_get_x(confidence_level=confidence_level)
    print(f"\nWe are {p_value * 100:,.2f}% confident that x will be less than {int(x)}. \n")


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
    test_ecdf_helper()
