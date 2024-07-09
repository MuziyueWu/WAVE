import scipy.optimize as opt
import numpy as np
import math

class modelValues():
    amp = None
    center_x = None
    center_y = None
    sigma_x = None
    sigma_y = None
    offset = None
    signal = None
    norm_height_guess = None
    error = None
    
def func(x_data_tuple, amplitude, x0, y0, sigma_x, sigma_y, offset):

    (x, y) = x_data_tuple
    y0 = float(y0)
    x0 = float(x0)
    I = offset + amplitude*np.exp( - (((x-x0)/sigma_x)**2 + ((y-y0)/sigma_y)**2)/2)
    
    return I.ravel()



def fit(func, local_tuple, data, guesses, bound_tup):
        param = modelValues()
        try:
            popt, pcov = opt.curve_fit(func, local_tuple, data.ravel(), p0 = guesses, bounds=bound_tup)
           
            param.amp = popt[0]
            param.center_x = popt[1]
            param.center_y = popt[2]
            param.sigma_x = popt[3]
            param.sigma_y = popt[4]
            param.offset = popt[5]
            param.signal = 2 * math.pi *  param.amp * param.sigma_x * param.sigma_y
            param.norm_height_guess = (param.amp + param.offset) / param.offset                                    
            param.error = np.sqrt(np.diag(pcov))
           
        except RuntimeError:
            print('RuntimeError')
            param.signal = 0
            param.amp = 0
            param.norm_height_guess = 0
            param.center_x = np.nan
            param.center_y = np.nan
            param.sigma_x = 5
            param.sigma_y = 4
            param.offset = guesses[5]
            

        except ValueError:
            print('ValueError')
            param.signal = 0
            param.amp = 0
            param.norm_height_guess = 0
            param.center_x = np.nan
            param.center_y = np.nan
            param.sigma_x = 4
            param.sigma_y = 4
            param.offset = guesses[5]
            

        return param
    
    
def getRSquared(func, data, par, size):
    """ Determine R^2 value for data and fit 2D Gaussian """
    x_val = np.linspace(0, size-1, size)
    y_val = np.linspace(0, size-1, size)
    x_val,y_val = np.meshgrid(x_val,y_val);
    I = func((x_val, y_val), par.amp, par.center_x, par.center_y, par.sigma_x, par.sigma_y, par.offset)
    ss_tot = ((data - np.mean(data))**2).sum()
    ss_res = ((data - I.reshape((size, size)))**2).sum()
    Rsqr = 1 - (ss_res/ss_tot)
    
    return (Rsqr, ss_res, ss_tot)



               