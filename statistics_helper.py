import pandas as pd
import numpy as np
import scipy.stats
import statistics
import math


def mean_confidence_interval(data, confidence=0.95):
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
    a, b = median_confidence_interval(data, cl)
    print(f"{cl*100}% Confidence Interval for Median is between {a:,.4f} and {b:,.4f}")


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
    test_median_confidence_interval()
