import numpy as np
from GUI_Pages.IntrinsicParametersFunctions.f_x_to_model_intrinsic import f_x_to_model_intrinsic
def calc_diff_intrinsic(im_real, x, seglen):
    # print(x)
    im_gray, _ = f_x_to_model_intrinsic(x, seglen, 0, im_real.shape[1], im_real.shape[0])
    im_gray = im_gray.astype(int)
    im_real = im_real.astype(int)
    # diff = np.clip( (im_gray - im_real), 0, 255) + np.clip( (im_real - im_gray), 0, 255 )
    diff = np.abs(im_gray - im_real) + np.abs(im_real - im_gray)

    diff = np.sum(diff.flatten())
    # diff.astype(np.float64)
    return diff

