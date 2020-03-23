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

def kn(x):
    E=511   #energia del gamma incidente
    ct = 1 - (511 / E) * (E/x -1)
    r = 2.81*10**(-13)    #cm
    a =r*r*((1 + ct**2) / 2)
    b = 1/((1 + E**2 *(1 - ct))**2)
    c =1 + ((E*(1-ct)**2) / ( (1 + ct**2)*(1 + E*(1-ct)) ) )
    sigma=a*b+c
    return sigma

x=np.linspace(511/3,511,1000)
y=kn(x)
m=np.max(y)

plt.plot(x,y)
plt.show()