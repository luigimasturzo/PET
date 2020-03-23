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
    #devo calibrare, voglio x in kev
    #per il primo caso approssimo e viene 
    #x=x*511/3.93
    a=x/511
    r=2.818     #fm
    b=(1-2*(a+1)/(a*a))*np.log(2*a+1)
    c=1/(2*(2*a +1)**2)
    sigma=np.pi*r*r/a*(b+0.5+4/a -c)
    ''' plt.figure()
        a=xdata[0]*511/3.93
        b=xdata[-1]*511/3.93
        di=int(b-a)
        xprova_histo=np.linspace(a,b,len(ydata))
        xprova=np.linspace(a,b,1000)
        #ydata,edges,__ = plt.hist(data_0,bins=100)
        rr=7.4*0.87*1.5
        plt.plot(xprova,rr*kn(xprova),label='KN')
        plt.plot(xprova_histo,ydata,label='data')
        plt.plot(xprova_histo,ydata-rr*kn(xprova_histo),label='y-kn')
        plt.xlabel('Energy [Kev]')
        plt.ylabel('c.s [fm]')
        plt.legend()
        plt.show()'''
    return sigma

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
        popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)
    elif ( _ == 1):
        '''popt,pcov = curve_fit(gauss, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gauss(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
        popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)
    else :
        '''popt,pcov = curve_fit(gauss, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gauss(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gauss,*popt)'''
        popt,pcov = curve_fit(gaussian_skew, xdata_fit, ydata_fit,p0=parametri)
        y_fit=gaussian_skew(x_fit,*popt)
        chiquadro=chi2(xdata_fit,ydata_fit,gaussian_skew,*popt)
    inc=np.sqrt(pcov.diagonal())
    
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
            #parametri=(100,4.7,0.5)         #gauss
            parametri=(100,4.7,0.5,-1)     #skew_gauss 
            #parametri =(50,3.93,0.5,50)     #gaussian KN
            first_extreme=3.5
            second_extreme=6.9
        elif ( _ == 1):
            #parametri=(100,3.17,0.5)        #gauss
            parametri=(100,3.17,0.5,-1)     #skew_gauss
            #parametri =(50,3.35,0.5,50)     #gaussian KN
            first_extreme=2.4
            second_extreme=4.3
        else :
            #parametri=(100,3.73,0.5)         #gauss
            parametri=(100,3.73,0.5,-1)     #skew_gauss 
            #parametri =(50,3.93,0.5,50)     #gaussian KN
            first_extreme=2.83
            second_extreme=4.87

        """
        Find straight line between extreme points and perform the fit.
        """
        m,q=linear(xdata,ydata,first_extreme,second_extreme)
        parametri=parametri + (m,) + (q,)
        popt,u,chiquadro=fit(xdata,ydata,first_extreme,second_extreme,parametri,_,item)

        R1 = 2.35*popt[2]/popt[1]
        R2=0
        uR1=2.35*np.sqrt((u[2]/popt[1])**2 + (popt[2]*u[1]/(popt[1]**2))**2)
        uR2=0

        logging.info('R = {} , {} , {}'.format(R1,R2, chiquadro))

        f.write('{} \t{} \t{} \t{} \t{} \t{} \t \n '.format(item,R1,uR1,R2,uR2,chiquadro))
    f.close()