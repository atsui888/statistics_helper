from typing import Union, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from numpy import asarray, exp
import scipy.stats
import statistics
import math
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF as statsmodels_ecdf
from sklearn.neighbors import KernelDensity
from scipy.stats import lognorm, weibull_min


class StatisticsHelper:
    """
    for usage, see the def test_ABC functions
    """
    def __init__(self, sample: Union[pd.Series, np.array], display_precision=2):
        self._sample = sample
        if self._sample is None or len(self._sample) < 1:
            raise SystemError("Sample Data cannot be None or 0 length")
        self.display_prec = display_precision
        self._sample_mean = self._sample.mean()
        self._sample_median = self._sample.median()
        self._sample_mode = self._sample.mode().values

        # the values in self._fitted_dist are returned from .fit() functions
        self._fitted_dist = None

    @property
    def sample(self):
        return self._sample

    def sample_mean(self):
        return self._sample_mean

    def sample_median(self):
        return self._sample_median

    def sample_mode(self):
        return self._sample_mean

    def __str__(self):
        msg = f'sample mean:\t{round(self._sample_mean, self.display_prec)}'
        msg += f'\nsample median:\t{round(self._sample_median, self.display_prec)}'
        mode = [round(i, self.display_prec) for i in self._sample_mode]
        msg += f'\nsample mode:\t{mode}'
        msg += f'\nIs distribution fitted?: {self._fitted_dist.get("fitted")}'
        return msg

    def fitted_dist(self):
        return self._fitted_dist

    def fit(self):
        raise NotImplementedError('Child Class needs to implement "fit" method.')

    def population_mean_confidence_interval(self, confidence=0.95):
        # https://pythonguides.com/scipy-confidence-interval/
        n = self._sample.shape[0]
        m = np.mean(self._sample)
        se = scipy.stats.sem(self._sample)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m-h, m+h

    def population_median_confidence_interval(self, confidence=0.95):
        dx = self._sample.sort_values(ascending=True, ignore_index=True)
        factor = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
        factor *= math.sqrt(len(dx))

        lower_interval = round(0.5 * (len(dx) - factor))
        upper_interval = round(0.5 * (1 + len(dx) + factor))

        return dx[lower_interval], dx[upper_interval]

    def plot_hist(self, fig_size=(8, 6), density=True, bins='auto', alpha=1, kde=False,
                  chart_title='Chart Title', x_label='Independent Var', y_label=None):
        """

        :param fig_size:
        :param density:
        :param bins:
        :param alpha:
        :param kde: Non-Parametric, don't show by default, as it is only relevant if we are fitting with a
            non-parametric distribution e.g. ECDF
        :param chart_title:
        :param x_label:
        :param y_label:
        :return:
        """
        if y_label is None and density:
            y_label = 'Probability Density'
        elif y_label is None and not density:
            y_label = 'Frequency'

        fig, ax = plt.subplots(figsize=fig_size)

        ax.hist(self._sample, density=density, bins=bins, alpha=alpha)
        plt.title(chart_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if kde:
            kde = scipy.stats.gaussian_kde(self._sample)
            xx = np.linspace(self._sample.min(), self._sample.max(), len(self._sample))
            ax.plot(xx, kde(xx))  # plot the PDF
        plt.show()


class NonParametricCDF(StatisticsHelper):
    """
    # https://www.mathworks.com/help/stats/nonparametric-and-empirical-probability-distributions.html
    # https://blockgeni.com/using-an-empirical-distribution-function-in-python/

    In some situations, you cannot accurately describe a data sample using a parametric distribution and cannot easily
    force the data into an existing distribution by doing data transforms or parameterization of the
    distribution function.

    In such cases, the Probability Density Function (pdf) or Cumulative Distribution Function (cdf) can be
    estimated from the data.

    The PDF returns the expected probability for observing a value. P(X=N)
        For discrete data, the PDF is referred to as a Probability Mass Function (PMF)

    The CDF returns the expected probability for observing a value less than or equal to a given value. P(X<=N)
    ---

    """
    def __init__(self, sample: Union[pd.Series, np.array]):
        super().__init__(sample)

    def fit(self):
        self._fitted_dist = statsmodels_ecdf(self._sample)

    def cdf(self, x):
        return self._fitted_dist(x)

    def plot_cdf(self, fig_size=(8, 6), alpha=1, chart_title='ECDF', x_label='Independent Var',
                 y_label='Cumulative Probability Density'):
        plt.plot(self._fitted_dist.x, self._fitted_dist.y)
        plt.title(chart_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


class NonParametricPDF(StatisticsHelper):
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html

    # https://www.mathworks.com/help/stats/nonparametric-and-empirical-probability-distributions.html
    # https://blockgeni.com/using-an-empirical-distribution-function-in-python/

    The PDF returns the expected probability for observing a value. P(X=N)
        For discrete data, the PDF is referred to as a Probability Mass Function (PMF)

    """
    def __init__(self, sample: Union[pd.Series, np.array], bandwidth=2, kernel='gaussian'):
        super().__init__(sample)
        # kernel{‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}, default=’gaussian’
        self._bandwidth = bandwidth
        self._kernel = kernel
        self._model = None
        self._probabilities = None
        self._x_values = None

    def fit(self):
        data = self._sample.values.reshape(-1, 1)

        self._model = KernelDensity(bandwidth=self._bandwidth, kernel=self._kernel)
        self._model.fit(data)

        self._x_values = asarray([value for value in range(int(data.min()), int(data.max()))])
        self._x_values = self._x_values.reshape(-1, 1)
        self._probabilities = self._model.score_samples(self._x_values)
        self._probabilities = exp(self._probabilities)

    def pdf(self, x):
        return self._probabilities[x]

    def cdf(self, x):
        # rc: funny ... if I have pdf, I can get the cdf right ?
        # since cdf is cumulative probabilities .. , correct?
        return self._probabilities[:x]

    def plot_pdf(self, fig_size=(8, 6), alpha=1, chart_title='PDF', x_label='Independent Var',
                 y_label='Probability Density'):
        plt.hist(self._sample, bins='auto', density=True)
        plt.plot(self._x_values[:], self._probabilities)

        plt.title(chart_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

def test_NonParametricPDF_bimodel():
    data = load_test_data_bimodal()
    pdf = NonParametricPDF(data)
    pdf.fit()
    print(f"P(x=40) =  {round(pdf.pdf(40), 3)}")
    pdf.plot_pdf()


def test_NonParametricPDF_tht():
    # seems like sklearn kernel density is not good for THT but is good for bi-modal data distributions
    data = load_test_data_tht()
    # kernel{‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}, default=’gaussian’
    pdf = NonParametricPDF(data, kernel='cosine')
    pdf.fit()
    print(f"P(x=200) = {round(pdf.pdf(200), 3)}")
    pdf.plot_pdf()
    pdf.plot_hist(kde=True)  # but why this kde is ok..


class Gamma(StatisticsHelper):
    # todo: Gamma Distribution unfinished
    def __init__(self, sample: Union[pd.Series, np.array]):
        super().__init__(sample)
        self._a_fit = None
        self._loc_fit = None
        self._scale_fit = None

    def fit(self):
        self._a_fit, self._loc_fit, self._scale_fit = scipy.stats.gamma.fit(self._sample)
        self._fitted_dist = scipy.stats.gamma(self._a_fit, self._loc_fit, self._scale_fit)

    def cdf(self, x):
        return self._fitted_dist.cdf(x)

    def plot_hist(self, fig_size=(8, 6), density=True, bins='auto', alpha=1, kde=False,
                  chart_title='Chart Title', x_label='Independent Var', y_label=None):
        # todo: add plot
        pass

    def plot_cdf(self):
        pass


def test_gamma_tht(show_charts=False):
    print('THT - Gamma Distribution')
    data = load_test_data_tht()
    gamma = Gamma(data, bandwidth=3)
    gamma.fit()
    x_values = [200, 250, 500, 750]
    for x in x_values:
        print(f"P(x<{x} = {gamma.cdf(x):.03f})")


def test_ecdf_tht(show_charts=False):
    print('THT - ECDF')
    data = load_test_data_tht()
    ecdf = NonParametricCDF(data)
    ecdf.fit()
    x_label = 'Total Handling Time (seconds)'
    if show_charts:
        ecdf.plot_hist(chart_title='Total Handling Time', x_label=x_label, kde=True)
    x_values = [200, 250, 500, 750]
    for x in x_values:
        print(f"P(x<{x} = {ecdf.cdf(x):.03f})")
    if show_charts:
        ecdf.plot_cdf(chart_title='Total Handling Time - CDF', x_label=x_label)


def test_ecdf_bimodal(show_charts=False):
    data = load_test_data_bimodal()
    ecdf = NonParametricCDF(data)
    ecdf.fit()
    x_label = 'Total Handling Time (seconds)'
    if show_charts:
        ecdf.plot_hist(chart_title='Bimodal Sample Data', x_label=x_label, kde=True)

    x_values = [20, 40, 60]
    for x in x_values:
        print(f"P(x<{x} = {ecdf.cdf(x):.03f})")
    if show_charts:
        ecdf.plot_cdf(chart_title='Bimodal Sample Data - CDF', x_label=x_label)



def load_test_data_tht():
    # tht = Total Handling Time (in a call center, per call)
    dft = pd.read_excel('DA-LSS.xlsx', sheet_name='THT', skiprows=8)
    cols = ['tht', 'training', 'unknown']
    dft.columns = cols
    dft.drop(columns='unknown', inplace=True)
    dft = dft['tht'].copy()
    return dft


def load_test_data_bimodal():
    from numpy.random import normal
    from numpy import hstack
    # generate a sample
    np.random.seed(42)
    sample1 = normal(loc=20, scale=5, size=300)
    sample2 = normal(loc=40, scale=5, size=700)
    sample = hstack((sample1, sample2))
    sample = pd.Series(sample)
    return sample


def test_population_mean_confidence_interval(data: Union[pd.Series, np.array], cl=0.95):
    sh = StatisticsHelper(data)
    print(sh)
    m, m_l, m_h = sh.population_mean_confidence_interval(confidence=cl)
    print(f"{m_l:,.0f}, {m:,.0f}, {m_h:,.0f}")
    print(f"{cl * 100}% Confidence that the Population Mean is between {m_l:,.0f} and {m_h:,.0f}")


def test_population_median_confidence_interval(data: Union[pd.Series, np.array], cl=0.95):
    sh = StatisticsHelper(data)
    low, high = sh.population_median_confidence_interval(confidence=cl)
    print(f"{cl * 100}% Confidence for Population Median is between {low:,.0f} and {high:,.0f}")


def test_plot_hist_tht():
    data = load_test_data_tht()
    sh = StatisticsHelper(data)
    # fig_size=(8, 6), density=True, bins='auto', alpha=1, kde=False
    sh.plot_hist(kde=True)


def test_plot_hist_bimodal():
    data = load_test_data_bimodal()
    sh = StatisticsHelper(data)
    # fig_size=(8, 6), density=True, bins='auto', alpha=1, kde=False
    sh.plot_hist(kde=True)



if __name__ == "__main__":

    # import sys
    # sys.path.insert(0, r'C:\Users\richa\OneDrive\RChai\Documents\00_Common_Code')
    # from statistics_helper import StatsHelper

    # load sample data - tht
    # TOTAL_HANDLING_TIME = load_test_data_tht()
    # test_population_mean_confidence_interval(TOTAL_HANDLING_TIME, cl=0.95)
    # test_population_median_confidence_interval(TOTAL_HANDLING_TIME, cl=0.95)
    # test_plot_hist(TOTAL_HANDLING_TIME)
    # test_plot_hist_tht()
    # test_plot_hist_bimodal()

    # THT - Right Skewed distribution
    # test_ecdf_bimodal()
    # test_ecdf_tht(show_charts=False)
    print()
    # test_gamma_tht(show_charts=False)
    print()
    test_NonParametricPDF_bimodel()
    # test_NonParametricPDF_tht()


