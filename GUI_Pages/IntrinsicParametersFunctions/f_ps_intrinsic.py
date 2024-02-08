import numpy as np
import scipy.optimize
import scipy
from GUI_Pages.IntrinsicParametersFunctions.calc_diff_intrinsic import calc_diff_intrinsic
from scipy.optimize import least_squares
from scipy.optimize import minimize
def f_ps_intrinsic(im0,x0,lb,ub,ims,mms,tm,noise,seglen):
    ObjectiveFunction = lambda x: (calc_diff_intrinsic(im0, x, seglen))

    xlb = x0 + lb + noise
    xub = x0 + ub + noise
    x0 += noise

    # xlb = x0 + lb
    # xub = x0 + ub
    # The current function does not allow us to have the same lower and upper bounds
    equal_mask = xlb == xub
    # if len(xub[equal_mask]) > 0: xub[equal_mask] += 10**(-10)

    if len(xub[equal_mask]) > 0: xub[equal_mask] += 10**(-1)


    x0 = x0.astype(np.float64)
    xlb = xlb.astype(np.float64)
    xub = xub.astype(np.float64)
    x0 = tuple(list(x0))
    xlb = tuple(list(xlb))
    xub = tuple(list(xub))

    # x0 *= 10
    # xlb *= 10
    # xub *= 10
    bounds = scipy.optimize.Bounds(xlb, xub)
    # bounds = (xlb, xub)
    print('about to compute')
    # res = least_squares(ObjectiveFunction, x0 + noise, method='trf', bounds = bounds )
    # res = least_squares(ObjectiveFunction, x0 , method='dogbox', bounds = bounds )
    res = minimize(ObjectiveFunction, x0, method='nelder-mead', bounds= bounds)
    return res.x, res.fun







