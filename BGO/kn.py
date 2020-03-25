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

def kn(x):
    E=511   #energia del gamma incidente
    ct = 1 - (511 / E) * (E/x -1)
    r = 2.81*10**(-13)    #cm
    a =r*r*((1 + ct**2) / 2)
    b = 1/((1 + E**2 *(1 - ct))**2)
    c =1 + ((E*(1-ct)**2) / ( (1 + ct**2)*(1 + E*(1-ct)) ) )
    sigma=a*b+c
    return sigma

def fitfinale(x, C, mu, sigma, A):
    return C * norm.pdf(x, mu, sigma) + A*kn(x)


def chi2(xdata,ydata,f,*popt):
    """
    This function returns chi squared value.
    """
    mask = ydata > 0
    chi2 = sum(((ydata[mask] - f(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2.)
    nu = mask.sum() - len(popt)
    sigma = np.sqrt(2 * nu)
    print('chi_square nrm    = {}'.format(chi2/nu))
    #print('expected value = {} +/- {}'.format(nu,sigma))
    return chi2/nu



n=400

data_0 = np.loadtxt('BGO1.F18.txt')
plt.title('Spectrum of {}'.format('BGO1.F18.txt'))
ydata,edges,__ = plt.hist(data_0,bins=n)
xdata = 0.5 * (edges[1:] + edges[:-1])
plt.grid()
plt.xlabel('Energia [a.u]')
plt.ylabel('Eventi [a.u]')

first_extreme=2.9
second_extreme=4.2
_b=np.where(xdata>first_extreme)
_c=np.where(xdata>second_extreme)
xdata_fit=xdata[_b[0][0]:_c[0][0]-1]
ydata_fit=ydata[_b[0][0]:_c[0][0]-1]
x_fit=np.linspace(xdata_fit[0],xdata_fit[-1],1000)


parametri=(100,4.7,0.5,210) 
popt,pcov = curve_fit(fitfinale, xdata_fit, ydata_fit,p0=parametri)
y_fit=fitfinale(x_fit,*popt)
chiquadro=chi2(xdata_fit,ydata_fit,fitfinale,*popt)

inc=np.sqrt(pcov.diagonal())
print(2.35*popt[2]/popt[1])




#mask=y_fit >0
plt.figure()
plt.title('Spectrum of {}, method gauss + KN'.format('BGO1.F18.txt'))
ydata,edges,_ = plt.hist(data_0,bins=n,label='Histo')
plt.plot(x_fit,y_fit,label='Fit')
plt.xlabel('Energia [a.u]')
plt.ylabel('Eventi [a.u]')
#x_pos = 0.1
#y_pos = 100
#c=2.35*popt[2]/popt[1]
#plt.text(x_pos, y_pos, '$\mu$ = {0:03}, $\sigma$ = {0:03}, R = {0:03}'.format(popt[1],popt[2],c))
plt.grid()
plt.legend()
plt.show()