import scipy.optimize as opt
import numpy as np
import math

class modelValues():
    a = None
    b = None
    r = None
    error = None

def func(data, a, b, r):
    x, y = data
    I = np.zeros(x.shape)
    a = float(a)
    b = float(b)

    array2D = np.power(x - a, 2) + np.power(y - b, 2)
    ringPixels = array2D <= r**2;
    I[ringPixels] = 1

    return I.ravel()
   


def fit(func, local_tuple, data, guesses, bound_tup):
        param = modelValues()
        try:
            popt, pcov = opt.curve_fit(func, local_tuple, data.ravel(), p0 = guesses, bounds=bound_tup)
           
            param.a = popt[0]
            param.b = popt[1]
            param.r = popt[2]                                
            param.error = np.sqrt(np.diag(pcov))
           
        except RuntimeError:
            print('RuntimeError')
            param.a = 10
            param.b = 10
            param.r = 10
            

        except ValueError:
            print('ValueError')
            param.a = 10
            param.b = 10
            param.r = 10
            

        return param
    
    
def getRSquared(func, data, par, size):
    """ Determine R^2 value for data and fit 2D Gaussian """
    x_val = np.linspace(0, size-1, size)
    y_val = np.linspace(0, size-1, size)
    x_val,y_val = np.meshgrid(x_val,y_val);
    I = func((x_val, y_val), par.a, par.b, par.r)
    ss_tot = ((data - np.mean(data))**2).sum()
    ss_res = ((data - I.reshape((size, size)))**2).sum()
    Rsqr = 1 - (ss_res/ss_tot)
    
    return (Rsqr, ss_res, ss_tot)



               