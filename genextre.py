import numpy as np
import math
import thinkstats2
import thinkplot

from stats_build import StatsBuild
from scipy.stats import genextreme
from lmoments3.distr import gev

class Gev(StatsBuild):

    estimadores = ['mvs', 'mml']

    def __init__(self, data=None,  shape=None, loc=None, scale=None):
        if data is None:
            if shape is None or loc is None or scale is None:
                raise ValueError("Parâmetros não  informados")
            else:
                self.shape = shape
                self.loc = loc
                self.scale = scale
        else:
            self.data = data

    def mml(self):
        if self.data is None:
            raise ValueError("Data not's None")
        mml = gev.lmom_fit(self.data)
        self.shape = mml['c']
        self.loc = mml['loc']
        self.scale = mml['scale']

        return self.shape, self.loc, self.scale

    def mvs(self):
        if self.data is None:
            raise ValueError("Data not's None")
        mvs = genextreme.fit(self.data)
        self.shape = mvs[0]
        self.loc = mvs[1]
        self.scale = mvs[2]

        return self.shape, self.loc, self.scale

    def prob(self, x, estimador):
        try:
            return genextreme.cdf(x, c=self.shape, loc=self.loc, scale=self.scale)
        except AttributeError:
            if estimador not in self.estimadores:
                raise ValueError('Estimador não existe')
            else:
                eval('self.' + estimador)()
            return self.prob(x, estimador=estimador) 

    def value(self, p, estimador=None):
        try:
            return genextreme.ppf(p, c=self.shape, loc=self.loc, scale=self.scale)
        except AttributeError:
            if estimador not in self.estimadores:
                raise ValueError('Estimador não existe')
            else:
                eval('self.' + estimador)()
            return self.value(p, estimador=estimador)
        

    def interval(self, alpha):
        inteval = genextreme.interval(alpha, c=self.shape, loc=self.loc, scale=self.scale)
        return inteval

    def MeanError(self, estimates, actual):
        """Computes the mean error of a sequence of estimates.

        estimate: sequence of numbers
        actual: actual value

        returns: float mean error
        """
        errors = [estimate-actual for estimate in estimates]
        return np.mean(errors)


    def RMSE(self, estimates, actual):
        """Computes the root mean squared error of a sequence of estimates.

        estimate: sequence of numbers
        actual: actual value

        returns: float RMSE
        """
        e2 = [(estimate-actual)**2 for estimate in estimates]
        mse = np.mean(e2)
        return math.sqrt(mse)

    
    def SimulateSample(self, n=9, m=1000):
        """Plots the sampling distribution of the sample mean.

        mu: hypothetical population mean
        sigma: hypothetical population standard deviation
        n: sample size
        m: number of iterations
        """
        def VertLine(x, y=1):
            thinkplot.Plot([x, x], [0, y], color='0.8', linewidth=3)

        means = []
        for _ in range(m):
            xs = genextreme.rvs(c=self.shape, loc=self.loc, scale=self.scale, size=n)
            xbar = np.mean(xs)
            means.append(xbar)

        stderr = self.RMSE(means, self.loc)
        print('Erro Padrão', stderr)

        cdf = thinkstats2.Cdf(means)
        ci = cdf.Percentile(5), cdf.Percentile(95)
        print('Intervalo de Confiança: ', ci)
        VertLine(ci[0])
        VertLine(ci[1])

        # plot the CDF
        thinkplot.Cdf(cdf)
        #thinkplot.Save(root='estimation1',
         #              xlabel='sample mean',
          #             ylabel='CDF',
           #            title='Sampling distribution')