"""
provo a fare il fit solo con KN
"""
import numpy as np
import os
import logging
import argparse
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.special import erf
from scipy import signal
logging.basicConfig(level=logging.INFO)


def gauss(x, C, mu, sigma, m, q):
    '''
    Gaussian function with linear offset.
    '''
    return C * norm.pdf(x, mu, sigma) - m*x + q

def kn(x,t):
    E=511   #energia del gamma incidente
    ct = 1 - (511 / E) * (E/x -1)
    r = 2.81*10**(-13)    #cm
    a =r*r*((1 + ct**2) / 2)
    b = 1/((1 + E**2 *(1 - ct))**2)
    c =1 + ((E*(1-ct)**2) / ( (1 + ct**2)*(1 + E*(1-ct)) ) )
    sigma=a*b+c
    return t*sigma

def fitfinale(x, C, mu, sigma, A):
    return C * norm.pdf(x, mu, sigma) + A*kn(x)





n=400

data_0 = np.loadtxt('BGO1.F18.txt')
plt.title('Spectrum of {}'.format('BGO1.F18.txt'))
ydata,edges,__ = plt.hist(data_0,bins=n)
xdata = 0.5 * (edges[1:] + edges[:-1])
plt.grid()
plt.xlabel('Energia [a.u]')
plt.ylabel('Eventi [a.u]')

d=511-170
x_fit=np.linspace(170,511,d)
popt,pcov=curve_fit(kn,x_fit,ydata[170:511],p0=[210])
plt.plot(xdata,kn(xdata,*popt))
plt.show()