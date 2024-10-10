import numpy as np
import jax
import time
import VBBinaryLensing
def timeit(f,iters=10,verbose=True):
    """
    A decorator to measure the execution time of a function, especially for JAX functions.

    This decorator calculates the mean and standard deviation of the execution time
    of the decorated function over a specified number of iterations.

    Args:
        f (callable): The function to be timed.
        iters (int, optional): The number of iterations to run the function for timing. Default is 10.

    Returns:
        callable: A wrapped function that, when called, prints the compile time and the mean 
                    and standard deviation of the execution time over the specified iterations.
    """
    def timed(*args,**kw):
        ts=time.perf_counter()
        result=f(*args,**kw)
        te=time.perf_counter()
        if verbose:
            print(f'{f.__name__} compile time={te-ts}')
        alltime=[]
        for i in range(iters):
            ts=time.perf_counter()
            result=f(*args,**kw)
            jax.block_until_ready(result)
            te=time.perf_counter()
            alltime.append(te-ts)
        alltime=np.array(alltime)
        if verbose:
            print(f'{f.__name__} time={np.mean(alltime)}+/-{np.std(alltime)}')
        return result,np.mean(alltime)
    return timed

def VBBL_light_curve(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol=0.,tol=1e-2):
    """
    Calculate the light curve of a binary lensing event using the VBBL model. Modified to the same coordinate system as the JAX model.

    Args:
        t_0 (float): The closest approach time. 
        u_0 (float): The impact parameter of the event.
        t_E (float): The Einstein crossing time.
        rho (float): The angular source size in the unit of the Einstein radius.
        q (float): The mass ratio of the binary lens.
        s (float): The separation of the binary lens in the unit of the Einstein radius.
        alpha_deg (float): The angle of the source trajectory in degrees.
        times (array): The times at which to calculate the light curve.
        retol (float): The relative tolerance. Default is 0.
        tol (float, optional): The tolerance. Default is 1e-2.
    
    Returns:
        array: The magnification of this parameter set. 
    """
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=np.pi+alpha_deg/180*np.pi
    VBBL.Tol=tol
    VBBL.RelTol=retol
    VBBL.BinaryLightCurve
    times=np.array(times)
    tau=(times-t_0)/t_E
    y1 = -u_0*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = u_0*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), u_0, alpha_VBBL, np.log(rho), np.log(t_E),t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return VBBL_mag

