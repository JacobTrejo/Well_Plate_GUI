import numpy as np
from IntrinsicParametersFunctions.calc_diff_intrinsic import calc_diff_intrinsic
from IntrinsicParametersFunctions.f_ps_intrinsic import f_ps_intrinsic

def f_fitmodel_intrinsic(x0, seglen, im_real):

    r = np.array([1, 1, 0.2, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15])
    R = np.array([r, r, r, r, r * 0.1])
    temp = np.random.random((5, 19))
    temp -= .5
    noise = temp * R
    noise[:, 2:] = 0
    lb = np.array([-10, -10, -0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ub = np.array([10, 10, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    x_t = np.zeros((5, 19))
    fval_t = np.zeros((5, 1))
    fval_t[0, 0] = calc_diff_intrinsic(im_real, x0, seglen)
    x_t[0, :] = x0

    print('Round 1: Tuning centroid and heading angle')

    for m in range(1,5):
        [x_t[m,:], fval_t[m,:]] = f_ps_intrinsic(im_real, x0, lb, ub, 0.2, 0.8, 0.005, noise[m,:], seglen)

    min_idx = np.argmin(fval_t)
    fval = fval_t[min_idx,:]
    x = x_t[min_idx,:]
    x0[0:3] = x[0:3]

    print('Round 2: Tuning length and relative positions of anterior components')

    for n in range(0,4):
        r = np.array([1, 1, 0.2, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15])
      #   R = np.array([r, r, r, r * 0.1])
      #   temp = np.random.random((4, 19))
      #   temp -= .5
      #   noise = temp * R
      #   lb = np.array([0, 0, 0, - 0.2, -0.3, -0.3, -0.3, -0.05, -0.05, -0.05, 0, 0, -.15, 0, -0.15, -0.15, -.2, -.2, 0])
      # # ub = np.array([0, 0, 0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.15, 0.4, 0.15, 0.15, .2, .3, 0])
      #   ub = np.array([0, 0, 0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.3, 0.8, 0.3, 0.3, .4, .6, 0])
      #
      #   x_t = np.zeros((5,19))
      #   x_t[4,:] = x
      #   # NOTE: should I rename temp
      #   fval_t = np.zeros((5,1))
      #   fval_t[4,:] = fval
      #   for m in range(0,4):
      #       [x_t[m,:], fval_t[m,:]] = f_ps_intrinsic(im_real, x0, lb, ub, 0.2, 0.8, 0.005, noise[m,:], seglen)
      #   min_idx = np.argmin(fval_t)
      #   fval = fval_t[min_idx]
      #   x = x_t[min_idx,:]
        amount = 1
        R = np.array([ r for _ in range(amount)])
        R[-1] *= .1
        temp = np.random.random((amount, 19))
        temp -= .5
        noise = temp * R
        lb = np.array([0, 0, 0, - 0.2, -0.3, -0.3, -0.3, -0.05, -0.05, -0.05, 0, 0, -.15, 0, -0.15, -0.15, -.2, -.2, 0])
        ub = np.array([0, 0, 0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.15, 0.4, 0.15, 0.15, .2, .3, 0])
        # ub = np.array([0, 0, 0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.8, 0.3, 0.8, 0.3, 0.3, .4, .6, 0])

        x_t = np.zeros((amount + 1, 19))
        x_t[amount, :] = x
        # NOTE: should I rename temp
        fval_t = np.zeros((amount + 1, 1))
        fval_t[amount, :] = fval
        for m in range(0, amount):
            [x_t[m, :], fval_t[m, :]] = f_ps_intrinsic(im_real, x0, lb, ub, 0.2, 0.8, 0.005, noise[m, :], seglen)
        min_idx = np.argmin(fval_t)
        fval = fval_t[min_idx]
        x = x_t[min_idx, :]






    print('Round 3: Tuning brightness intrinsic parameters')
    for n in range(0,4):
        r = np.array([1, 1, .2, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15, .15])
        # R = np.array([r, r, r, r * 0.1])
        # temp = np.random.random((4, 19))
        # temp -= .5
        # noise = temp * R
        # lb = np.array([0, 0, 0, 0, 0, 0, 0, -0.05, -0.05, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # ub = np.array([0, 0, 0, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, .4])
        # x_t = np.zeros((5, 19))
        # x_t[4, :] = x
        # fval_t = np.zeros((5, 1))
        # fval_t[4, :] = fval
        # for m in range(0,4):
        #     [x_t[m,:], fval_t[m,:]] = f_ps_intrinsic(im_real, x0, lb, ub, 10, 0.8, 0.005, noise[m,:], seglen)
        # min_idx = np.argmin(fval_t)
        # fval = fval_t[min_idx]
        # x = x_t[min_idx, :]
        amount = 1
        R = np.array([r for _ in range(amount)])
        R[-1] *= .1
        temp = np.random.random((amount, 19))
        temp -= .5
        noise = temp * R
        lb = np.array([0, 0, 0, 0, 0, 0, 0, -0.05, -0.05, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ub = np.array([0, 0, 0, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, .4])
        x_t = np.zeros((amount + 1, 19))
        x_t[amount, :] = x
        # NOTE: should I rename temp
        fval_t = np.zeros((amount + 1, 1))
        fval_t[amount, :] = fval
        for m in range(0, amount):
            [x_t[m, :], fval_t[m, :]] = f_ps_intrinsic(im_real, x0, lb, ub, 0.2, 0.8, 0.005, noise[m, :], seglen)
        min_idx = np.argmin(fval_t)
        fval = fval_t[min_idx]
        x = x_t[min_idx, :]


    return x, fval









