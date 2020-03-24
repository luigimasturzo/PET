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
"""
When you change fit method you have to go to change commented lines in fit function and parameters.
"""

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
    Gaussian function with linear offset.
    '''
    return C * norm.pdf(x, mu, sigma) - m*x + q

def dg(x,a1,b1,c1,a2,b2,c2,m,q):
    """
    Double gaussian distribution with linear offset.
    """
    return a1*np.exp(-((x-b1)/c1)**2) + a2*np.exp(-((x-b2)/c2)**2) -m*x +q

def gaussian_skew(x,A,x0,sigma,alpha,m,q):   
    """
    gaussian-skew distribution with linear offset.
    """ 
    return gauss(x,A,x0,sigma,m,q)+erf(alpha*(x-x0)/sigma)

def kn(x):
    E=511   #energia del gamma incidente
    ct = 1 - (511 / E) * (E/x -1)
    r = 2.81*10**(-13)    #cm
    a =r*r*((1 + ct**2) / 2)
    b = 1/((1 + E**2 *(1 - ct))**2)
    c =1 + ((E*(1-ct)**2) / ( (1 + ct**2)*(1 + E*(1-ct)) ) )
    sigma=a*b+c
    return sigma

def fitfinale(x, C, mu, sigma, m, q, A):
    return C * norm.pdf(x, mu, sigma) - m*x + q + A*kn(x)


def chi2(xdata,ydata,f,*popt):
    """
    This function returns chi squared value.
    """
    mask = ydata > 0
    chi2 = sum(((ydata[mask] - f(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2.)
    nu = mask.sum() - len(popt)
    sigma = np.sqrt(2 * nu)
    print('chi_square     = {}'.format(chi2/nu))
    print('expected value = {} +/- {}'.format(nu,sigma))
    return chi2/nu

def linear(xdata,ydata,first_extreme,second_extreme):
    """
    Find angular coefficient and intercept of a straight line given two points.
    """
    _b=np.where(xdata>first_extreme)
    _c=np.where(xdata>second_extreme)
    xdata_line=xdata[_b[0][0]:_c[0][0]-1]
    ydata_line=ydata[_b[0][0]:_c[0][0]-1]
    m = (ydata_line[-1]-ydata_line[0])/(xdata_line[-1]-xdata_line[0])
    q = ((ydata_line[0]*xdata_line[-1]) - (xdata_line[0]*ydata_line[-1]))/(xdata_line[-1]-xdata_line[0])
    return m,q

def fit(xdata,ydata,first_extreme,second_extreme,parametri,_,item):
    """
    Function to control fit procedure.
    """
       
    _b=np.where(xdata>first_extreme)
    _c=np.where(xdata>second_extreme)
    xdata_fit=xdata[_b[0][0]:_c[0][0]-1]
    ydata_fit=ydata[_b[0][0]:_c[0][0]-1]
    x_fit=np.linspace(xdata_fit[0],xdata_fit[-1],1000)

    if ( _ == 0):
        '''popt,pcov = curve_fit(gauss, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gauss(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
        '''popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)'''
        popt,pcov = curve_fit(fitfinale, xdata_fit, ydata_fit,p0=parametri)
        y_fit=fitfinale(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,fitfinale,*popt)
        ''' popt,pcov = curve_fit(gaussian_KleinNishina, xdata_fit, ydata_fit,p0=parametri)
        y_fit_1=gaussian_KleinNishina(xdata_fit,*popt)
        y_fit=ydata_fit-gaussian_KleinNishina(xdata_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
    elif ( _ == 1):
        '''popt,pcov = curve_fit(gauss, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gauss(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
        '''popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)'''
        popt,pcov = curve_fit(fitfinale, xdata_fit, ydata_fit,p0=parametri)
        y_fit=fitfinale(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,fitfinale,*popt)
    elif ( _ == 2):
        popt,pcov = curve_fit(dg, xdata_fit, ydata_fit,p0=parametri)
        y_fit=dg(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,dg,*popt)
    elif ( _ == 3):
        '''popt,pcov = curve_fit(gauss, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gauss(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
        '''popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)'''
        popt,pcov = curve_fit(fitfinale, xdata_fit, ydata_fit,p0=parametri)
        y_fit=fitfinale(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,fitfinale,*popt)
    else :
        popt,pcov = curve_fit(dg, xdata_fit, ydata_fit,p0=parametri)
        y_fit=dg(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,dg,*popt)
    inc=np.sqrt(pcov.diagonal())
    print('000 -> A = {}'.format(popt[3]))
    
    if args.show is not None:
    
        #mask=y_fit >0
        plt.figure()
        plt.title('Fit of {} spectrum'.format(item))
        ydata,edges,_ = plt.hist(data_0,bins=100,label='Histo')
        plt.plot(x_fit,y_fit,label='Fit')
        plt.xlabel('Energia [a.u]')
        plt.ylabel('Eventi [a.u]')
        plt.grid()
        plt.legend()
        plt.show()
    return popt,inc,chiquadro



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-s', '--show', help='Do you want to show the images?')
    args = parser.parse_args()

    f = open('results/risoluzione.txt', 'w') 
    f.write('{} \t{} \t{} \t{} \t{} \t{} \t \n '.format('Name','Res1','sigma(Res1)','Res2','sigma(Res2)','chi_square'))
    file_grezzi = glob.glob('*.csv')
    change_extension(file_grezzi)
    files = glob.glob('*txt') 
    for _, item in enumerate(files):
        print('----- {} -----'.format(item))
        print(_)

        """
        Load the file and obtain the histogram.
        """
        data_0 = np.loadtxt(item)
        plt.title('Spectrum of {}'.format(item))
        ydata,edges,__ = plt.hist(data_0,bins=100)
        xdata = 0.5 * (edges[1:] + edges[:-1])
        plt.grid()
        plt.xlabel('Energia [a.u]')
        plt.ylabel('Eventi [a.u]')
        plt.show()

        """
        Find the border channels (extremes) manually defining parameters.
        """
        if ( _ == 0):
            #parametri=(100,3.93,0.5)         #gauss
            #parametri=(100,3.93,0.5,-4)     #skew_gauss 
            #parametri =(50,3.93,0.5,50)     #gaussian KN
            parametri=(100,3.93,0.5)        #fit finale
            first_extreme=3.21
            second_extreme=4.65
        elif ( _ == 1):
            #parametri=(100,3.35,0.5)        #gauss
            #parametri=(100,3.35,0.5,-4)     #skew_gauss
            #parametri =(50,3.35,0.5,50)     #gaussian KN
            parametri=(100,3.35,0.5)        #fit finale
            first_extreme=2.59
            second_extreme=4.27
        elif ( _ == 2):
            parametri=(3140,2.486,0.1881,3100,2.175,0.492)
            first_extreme=1.61
            second_extreme=3.09
        elif ( _ == 3):
            #parametri=(100,2.76,0.5)        #gauss
            #parametri=(100,2.76,0.5,-2)     #skew_gauss
            #parametri =(50,2.76,0.5,50)     #gaussian KN
            parametri=(100,2.76,0.5)    #fitfinale
            first_extreme=2.44
            second_extreme=3.2
        else :
            parametri=(5766,2.504,0.1836,3007,2.061,0.2861)
            first_extreme=1.61
            second_extreme=3.09

        """
        Find straight line between extreme points and perform the fit.
        """
        m,q=linear(xdata,ydata,first_extreme,second_extreme)
        if(_ == 0 or _ == 1 or _ == 3):
            a=110
            parametri=parametri + (m,) + (q,) + (a,)
        else:
            parametri=parametri + (m,) + (q,)
        popt,u,chiquadro=fit(xdata,ydata,first_extreme,second_extreme,parametri,_,item)

        if (_ == 2 or _ == 4):
            R1=2.35*popt[5]/popt[4]
            R2=2.35*popt[2]/popt[1]
            uR1=2.35*np.sqrt((u[5]/popt[4])**2 + (popt[5]*u[4]/(popt[4]**2))**2)
            uR2=2.35*np.sqrt((u[2]/popt[1])**2 + (popt[2]*u[1]/(popt[1]**2))**2)
        else:
            R1 = 2.35*popt[2]/popt[1]
            R2=0
            uR1=2.35*np.sqrt((u[2]/popt[1])**2 + (popt[2]*u[1]/(popt[1]**2))**2)
            uR2=0

        logging.info('R = {} , {} , {}'.format(R1,R2, chiquadro))

        f.write('{} \t{} \t{} \t{} \t{} \t{} \t \n '.format(item,R1,uR1,R2,uR2,chiquadro))
    f.close()