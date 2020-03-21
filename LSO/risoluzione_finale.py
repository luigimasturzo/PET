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

_description = 'Analize spectrum and find energy resolution'

def change_extension(file_csv):
    """
    This function aloows you to change files' format from csv to txt.
    """
    if len(file_csv) != 0:    
        for i in range(0,len(file_grezzi),1):
            
            source = file_csv[i]
            filename, file_extension = os.path.splitext(source)
            dest = filename+'.txt'
            os.rename(source, dest)
            print('File correctly transformed from csv to txt format!')
    else:
        print('Files are already in txt format.')
    return

def gauss(x, C, mu, sigma, m, q):
    '''
    Gaussiana con offset lineare
    '''
    return C * norm.pdf(x, mu, sigma) - m*x +q

def linear(x1,x2,y1,y2,item):
    """
    Find angular coefficient and intercept of a straight line given two points.
    """
    m = (y2-y1)/(x2-x1)
    q = ((y1*x2) - (x1*y2))/(x2-x1)
    return m,q

files = glob.glob('*txt') 
data_0 = np.loadtxt(files[0])
ydata,edges,__ = plt.hist(data_0,bins=100)
xdata = 0.5 * (edges[1:] + edges[:-1])


first_extreme=3.21
second_extreme=4.65


_b=np.where(xdata>first_extreme)
_c=np.where(xdata>second_extreme)
xdata_line=xdata[_b[0][0]:_c[0][0]-1]
ydata_line=ydata[_b[0][0]:_c[0][0]-1]
m,q=linear(xdata_line[0],xdata_line[-1],ydata_line[0],ydata_line[-1],files[0])

parametri=(100,3.93,0.5,m,q)         #gauss


popt,pcov=curve_fit(gauss,xdata_line,ydata_line,p0=parametri)
plt.plot(xdata_line,gauss(xdata_line,*popt))
plt.show()
