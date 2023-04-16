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


class StatsHelper:
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
    .mean_confidence_interval(data, confidence=0.95)
        .test_mean_confidence_interval()
    .median_confidence_interval(data, confidence=.95)
        .test_median_confidence_interval()
    .plot_cdf()
    .get_cumulative_probability(x)
        .test_get_cumulative_probability(x)
    .with_p_value_get_x(p: float = 0.5):
        . test_with_p_value_get_x()

    methods - internal
    ------------------

    """

    def __init__(self, df: pd.Series):
        self._data = df
        self._fitted = False
        self._ecdf = None

    def fit(self, fit_type: str = 'ecdf'):
        """
        todo: add fit ['normal', 'lognormal', 'weibull']

        There are two main types of probability distribution functions:
                1. Probability Density Function (PDF).
                2. Cumulative Distribution Function (CDF).

            The PDF returns the expected probability for observing a value.
                P(x = n)
                i.e. the probability of one specific x-value occurring.
                For discrete data, the PDF is referred to as a Probability Mass Function (PMF).

            # https://www.youtube.com/watch?v=FhZdVPX1rf0
            # https://www.youtube.com/watch?v=Lktb-GADVs8
            The CDF returns the expected probability for observing a value less than or equal to a given value.
                P(x <= n)
                i.e. the probability of all x-values up to a certain x occurring.

                CDF
                . non-decreasing ("monotonic"), the height can never go down.
                    i.e. f(Xn) >= f(Xn-1)
                . as it approaches negative infinity, p approaches 0
                    x-axis can be negative e.g. negative temperatures, and y-axis (p-values) head towards direction
                        of 0
                . as it approaches positive infinity, p approaches 1

                The CDF adds up the area (under curve) of the PDF
                    i.e. if PMF (discrete)
                        e.g. F(x < 2) = f(x=0) + f(x=1)
                            where big F is CDF and little f is PDF
                        if PDF (continuous)
                            F(x < 2) = adding up the f(the infinite values of the areas before x=2)
            side note:
                if y axis can also be written as f(x), i.e. given a x value (on x-axis),
                    f(x) gives us the y-value (on y-axis)

        :param fit_type:
        :return:
        """
        if fit_type.lower() == 'ecdf':
            # https://machinelearningmastery.com/empirical-distribution-function-in-python/
            # ECDF is a non-parametric method (to confirm)
            """                           
            An "empirical cumulative distribution function" is called the "Empirical Distribution Function", 
            or `EDF` for short. 
            It is also referred to as the "Empirical Cumulative Distribution Function", or `ECDF`.   
            
            for details on how to calculate: 
                https://machinelearningmastery.com/empirical-distribution-function-in-python/
                search for text: "The EDF is calculated by ordering all of the unique observations in ..."
                and
                Probability distributions from empirical data | Probability & combinatorics
                    https://www.youtube.com/watch?v=wztjEa7893c
            """
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

    def with_interval_get_p_value(self, x1: float, x2: float):
        if not self._fitted:
            msg = "Please fit ECDF with data first, otherwise all values are None"
            print(f"\n{msg}")
            raise SystemError(msg)

        try:
            if x1 <= x2:
                msg = "The second number passed in must be larger than the first."
                print(msg)
                raise SystemError(msg)
        except Exception as e:
            print(e)

        p_x1 = self.get_cumulative_probability(x1)
        p_x2 = self.get_cumulative_probability(x2)
        return p_x1, p_x2, p_x2-p_x1

    def population_mean_confidence_interval(self, confidence=0.95):
        # https://pythonguides.com/scipy-confidence-interval/
        n = self._data.shape[0]
        m = np.mean(self._data)
        se = scipy.stats.sem(self._data)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m-h, m+h

    def population_median_confidence_interval(self, confidence=0.95):
        dx = self._data.sort_values(ascending=True, ignore_index=True)
        factor = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
        factor *= math.sqrt(len(dx))

        lower_interval = round(0.5 * (len(dx) - factor))
        upper_interval = round(0.5 * (1 + len(dx) + factor))

        return dx[lower_interval], dx[upper_interval]


# Testing Functions - below
def test_population_median_confidence_interval(confidence=0.95):
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    print(f"Sample Median = {int(df['tht'].median())}")

    zz = StatsHelper(df['tht'])

    low, high = zz.population_median_confidence_interval(confidence=confidence)
    print(f"{confidence * 100}% Confidence for Population Median is between {low:,.0f} and {high:,.0f}")


def test_population_mean_confidence_interval(confidence=0.95):
    """
    Given a sample, estimate the population mean
    :param confidence:
    :return:
    """
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    print(f"Sample Mean = {int(df['tht'].mean())}")

    zz = StatsHelper(df['tht'])
    m, m_l, m_h = zz.population_mean_confidence_interval(confidence=confidence)
    print(f"{m_l:,.0f}, {m:,.0f}, {m_h:,.0f}")
    print(f"{confidence * 100}% Confidence that the Population Mean is between {m_l:,.0f} and {m_h:,.0f}")


def test_with_interval_get_p_value(n1: float, n2: float, show_chart=False):
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    zz = StatsHelper(df['tht'])
    zz.fit()
    if show_chart:
        zz.plot_cdf()

    p_n1, p_n2, p_interval = zz.with_interval_get_p_value(x1=n1, x2=n2)
    print(f"\np_interval = {p_n2:,.2f} - {p_n1:,.2f} = {p_interval:,.2f}")
    print(f"P({n1} < X < {n2}) = {p_interval:,.2f}")


def test_get_cumulative_probability(x, show_chart=False):
    df = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    df.columns = cols
    df.drop(columns='unknown', inplace=True)
    zz = StatsHelper(df['tht'])
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

    zz = StatsHelper(df['tht'])
    zz.fit()
    if show_chart:
        zz.plot_cdf()
    # print(f"P(x<240): {zz.get_cumulative_probability(240):,.3f}")
    # print(f"P(x<661): {zz.get_cumulative_probability(661):,.3f}")
    p_value = p
    x_val, p = zz.with_p_value_get_x(desired_p=p_value)
    print(f"The probability of values being less than {x_val} is {p}.")



if __name__ == "__main__":
    # cl = 0.95
    # test_population_mean_confidence_interval(confidence=cl)
    # test_population_median_confidence_interval(confidence=cl)
    # test_median_confidence_interval()

    # # X = [500, 501, 699, 700]
    # What % of telephone calls are handled within 600s i.e. X=600
    X = 600
    test_get_cumulative_probability(X, show_chart=True)
    # 91% of calls are handled within 600s.

    # What service level can I promise within 95% of the handled calls?
    # i.e. p=0.95
    # p = 0.95
    # test_with_p_value_get_x(p=p, show_chart=False)
    # In 95% of the calls, they will on average be handled with 793 seconds
    # This is our current SLA capability.
    #
    # test_with_interval_get_p_value(n1=300, n2=350, show_chart=True)
    pass
