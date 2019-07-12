import calendar as cal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import thinkplot
import thinkstats2
import scipy.stats as ss
import random
import survival


def dados(dir, label):
    
    dados = pd.read_csv(dir, index_col=0, parse_dates=True)
    boolean = dados[label].isnull()
    date = boolean.loc[boolean == False].index

    return dados[label].loc[date[0]:date[-1]]

def date(dados, start, end):
    date_start = pd.to_datetime(start, dayfirst=True)
    date_end = pd.to_datetime(end, dayfirst=True)
    
    return dados.loc[date_start:date_end]

def month_start_year_hydrologic(dados):
    mean_month = [dados.loc[dados.index.month == i].mean() for i in range(1, 13)]
    month_start_year_hydrologic = 1 + mean_month.index(min(mean_month))
    month_start_year_hydrologic_abr = cal.month_abbr[month_start_year_hydrologic].upper()
    month_num = month_start_year_hydrologic
    month_abr = month_start_year_hydrologic_abr
    
    return month_num, month_abr

def annual(dados, month_start_year_hydrologic):
    data_by_year_hydrologic = dados.groupby(pd.Grouper(freq='AS-%s' % month_start_year_hydrologic[1]))
    max_vazao = data_by_year_hydrologic.max().values
    idx_vazao = data_by_year_hydrologic.idxmax().values

    peak = pd.Series(max_vazao, index=idx_vazao, name='VMA')

    return peak

def dados_media(dados, freq='M'):
    dados = dados.groupby(pd.Grouper(freq=freq)).mean()
    return dados

def dados_acumulado(dados, freq='M'):
    dados = dados.groupby(pd.Grouper(freq=freq)).sum()
    if freq == 'A':
        return dados.loc[dados.values > 1000]
    return dados

def plot_percentil(dados_chuva, dados_vazao):
    dados_month = pd.DataFrame([dados_chuva, dados_vazao])
    dados_month = dados_month.T
    
    dados_month = dados_month.dropna(subset=[dados_chuva.name, dados_vazao.name])
    bins = np.arange(0, 400, 40)
    indices = np.digitize(dados_month[dados_chuva.name], bins)
    groups = dados_month.groupby(indices)
    
    mean_chuva = [group[dados_chuva.name].mean() for i, group in groups]
    cdfs = [thinkstats2.Cdf(group[dados_vazao.name]) for i, group in groups]
    
    for percent in [75, 50, 25]:
        vazao_percentiles = [cdf.Percentile(percent) for cdf in cdfs]
        label = '%d°' % percent
        thinkplot.Plot(mean_chuva, vazao_percentiles, label=label)

def expon_distribution(dados):
    
    mu,sigma=ss.expon.fit(dados)
    lin = np.linspace(0, dados.max(), num=100)
    
    expon_cdf = ss.expon.cdf(lin, mu, sigma)
    plt.plot(lin, expon_cdf, color='black')
    
def normal_pdf(dados):
    
    mean = dados.mean()
    std_dev = dados.std()
    lin = ss.norm.rvs(size=10000, loc=mean, scale=std_dev) 
    x = np.sort(lin)
    normal = ss.norm.pdf(x, mean, std_dev)
    plt.plot(x, normal, color='black')

    
def normal_cdf(dados, label):
    
    mean = dados.mean()
    std_dev = dados.std()
    lin = ss.norm.rvs(size=10000, loc=mean, scale=std_dev) 
    x = np.sort(lin)
    normal = ss.norm.cdf(x, mean, std_dev)
    plt.plot(x, normal, color='black', label=label)
    
def lognormal_pdf(dados):
    
    s, loc, scale = ss.lognorm.fit(dados)
    print(s)
    lin = ss.lognorm.rvs(s=s, size=1000, loc=loc, scale=scale) 
    x = np.sort(lin)
    normal = ss.norm.pdf(x, loc=loc, scale=scale)
    plt.plot(x, normal, color='black')

    
def lognormal_cdf(dados, label):
    
    s, loc, scale = ss.lognorm.fit(dados)
    lin = ss.lognorm.rvs(s=s, size=1000, loc=loc, scale=scale)
    x = np.sort(lin)
    normal = ss.norm.cdf(x, loc=loc, scale=scale)
    plt.plot(x, normal, color='black', label=label)

    
def pareto_cdf(dados, label):

    b, loc, scale = ss.pareto.fit(dados)
    lin = ss.pareto.rvs(b=b, size=1000, loc=loc, scale=scale)
    x = np.sort(lin)
    pareto_cdf = ss.pareto.cdf(x, b=b, loc=loc, scale=scale)
    plt.plot(x, pareto_cdf, color='black', label=label)
    
def pareto_pdf(dados):
    
    b, loc, scale = ss.pareto.fit(dados)
    lin = ss.pareto.rvs(b=b, size=1000, loc=loc, scale=scale)
    x = np.sort(lin)
    pareto_pdf = ss.pareto.pdf(x, b=b, loc=loc, scale=scale)
    plt.plot(x, pareto_pdf, color='black')

def remove_outlier(dados):
    Q1 = dados.quantile(0.25)
    Q3 = dados.quantile(0.75)
    IQR = Q3 - Q1
    trueList = ~((dados < (Q1 - 1.5 * IQR)) |(dados > (Q3 + 1.5 * IQR)))
    return dados[trueList]

    
def plot_gev(shape, loc, scale):
    
    dist = ss.genextreme(c=shape, loc=loc, scale=scale)
    xs = dist.rvs(size=100)
    ys = dist.cdf(xs)
    plt.plot(np.sort(xs), np.sort(ys))
    return dist
    
class CoinTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

class DiffStdPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.std() - group2.std())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
    
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class DiceChiTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        observed, expected = data
        test_stat = sum((observed - expected)**2 / expected)
        return test_stat
    
    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class PregLengthTest(thinkstats2.HypothesisTest):
    
    def TestStatistic(self, data):
        observado, esperado = data
        stat = self.ChiSquared(observado) + self.ChiSquared(esperado)
        return stat

    def ChiSquared(self, lengths):
        hist = thinkstats2.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected)**2 / expected)
        return stat
    
    def MakeModel(self):
        observado, esperado = self.data
        self.n = len(observado)
        self.pool = np.hstack((observado, esperado))

        pmf = thinkstats2.Pmf(self.pool)
        self.values = observado
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

def SamplingDistributions(dados_chuva, dados_vazao, iters=101):
    dados = pd.DataFrame([dados_chuva, dados_vazao])
    dados = dados.T
    t = []
    for _ in range(iters):
        sample = thinkstats2.ResampleRows(dados)
        chuva = sample["COIMBRA_P"]
        vazao = sample["COIMBRA_F"]
        estimates = thinkstats2.LeastSquares(chuva, vazao)
        t.append(estimates)

    inters, slopes = zip(*t)
    return inters, slopes
        
def Summarize(estimates, actual=None):
    mean = thinkstats2.Mean(estimates)
    stderr = thinkstats2.Std(estimates, mu=actual)
    cdf = thinkstats2.Cdf(estimates)
    ci = cdf.ConfidenceInterval(90)
    print('mean: ', mean,  
          '\nSE: ',stderr, 
          '\nCI: ', ci)

def PlotConfidenceIntervals(xs, inters, slopes, percent=90, **options):
    fys_seq = []
    for inter, slope in zip(inters, slopes):
        fxs, fys = thinkstats2.FitLine(xs, inter, slope)
        fys_seq.append(fys)

    p = (100 - percent) / 2
    percents = p, 100 - p
    low, high = thinkstats2.PercentileRows(fys_seq, percents)
    thinkplot.FillBetween(fxs, low, high, **options)

class SlopeTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        chuva, vazao = data
        _, slope = thinkstats2.LeastSquares(chuva, vazao)
        return slope

    def MakeModel(self):
        _, vazao= self.data
        self.ybar = vazao.mean()
        self.res = vazao - self.ybar

    def RunModel(self):
        chuva, _ = self.data
        vazao = self.ybar + np.random.permutation(self.res)
        return chuva, vazao
    
def EstimateMarriageSurvival(dados, limiar, label=""):
    
    complete = dados[dados >= limiar].dropna()
    ongoing = dados[dados < limiar]

    hf = survival.EstimateHazardFunction(complete, ongoing, label=label)
    sf = hf.MakeSurvival()

    return hf, sf

def ResampleSurvival(dados, limiar, iters=101):
    """Resamples respondents and estimates the survival function.

    resp: DataFrame of respondents
    iters: number of resamples
    """ 
    _, sf = EstimateMarriageSurvival(dados, limiar)
    thinkplot.Plot(sf)

    low, high = dados.min(), dados.max()
    ts = np.arange(low, high, 1)

    ss_seq = []
    for _ in range(iters):
        sample = thinkstats2.ResampleRowsWeighted(pd.DataFrame(dados), column='MANSO')
        _, sf = EstimateMarriageSurvival(sample['MANSO'], limiar)
        ss_seq.append(sf.Probs(ts))
    
    low, high = thinkstats2.PercentileRows(ss_seq, [5, 95])
    thinkplot.FillBetween(ts, low, high, color='gray', label='90% CI')