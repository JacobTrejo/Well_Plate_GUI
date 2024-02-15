import numpy as np
from GUI_Pages.IntrinsicParametersFunctions.drawEllipsoid import drawEllipsoid
from GUI_Pages.IntrinsicParametersFunctions.subAuxilary import *
from GUI_Pages.IntrinsicParametersFunctions.project import project
from numpy.random import normal as normrnd

def draw_anterior_b_intrinsic(seglen, theta, gamma, phi, dh1, dh2, dimension, size_lut, randomize, intrinsic_params):
    d_eye = intrinsic_params[0] * seglen
    c_eyes = intrinsic_params[1]
    c_belly = intrinsic_params[2]
    c_head = intrinsic_params[3]
    eyes_br = intrinsic_params[4] * 1e3
    belly_br = intrinsic_params[5] * 1e3
    head_br = intrinsic_params[6] * 1e3
    eye_w = intrinsic_params[7] * seglen
    eye_l = intrinsic_params[8] * seglen
    belly_w = intrinsic_params[9] * seglen
    belly_l = intrinsic_params[10] * seglen
    head_w = intrinsic_params[11] * seglen
    head_l = intrinsic_params[12] * seglen

    XX = size_lut
    YY = size_lut
    ZZ = size_lut

    # # # Note: temporary
    # c_eyes = 1.7
    # c_belly = .9
    # c_head = 1.1
    canvas = np.zeros((XX, YY, ZZ))

    # Rotation matrix
    R = rotz(theta) @ roty(phi) @ rotx(0)

    # Initialize points of the ball and stick model in the canvas
    pt_original = np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:, 1] = np.array([np.ceil(XX / 2) + dh1, np.ceil(YY / 2) + dh2, np.ceil(ZZ / 2)])
    pt_original[:, 0] = pt_original[:, 1] - np.array([seglen, 0, 0], dtype=pt_original.dtype)
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0], dtype=pt_original.dtype)

    # Initialize centers of eyes, belly and head with respect to the ball and stick model
    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]

    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    belly_c = np.array([[c_belly * pt_original[0, 1] + (1 - c_belly) * pt_original[0, 2]],
                        [c_belly * pt_original[1, 1] + (1 - c_belly) * pt_original[1, 2]],
                        [pt_original[2, 1] - seglen / 6]], dtype=pt_original.dtype)
    belly_c = belly_c - pt_original[:, 1, None]
    belly_c = np.matmul(R, belly_c) + pt_original[:, 1, None]

    head_c = np.array([[c_head * pt_original[0, 0] + (1 - c_head) * pt_original[0, 1]],
                       [c_head * pt_original[1, 0] + (1 - c_head) * pt_original[1, 1]],
                       [np.ceil(XX / 2) - seglen / 6]], dtype=pt_original.dtype)
    head_c = head_c - pt_original[:, 1, None]
    head_c = np.matmul(R, head_c) + pt_original[:, 1, None]

    eye1_br = eyes_br
    eye2_br = eyes_br

    rand3_eye = randomize * normrnd(1, 0.05) + (1 - randomize)

    eye_h = seglen * 0.3

    belly_h = seglen * 0.36

    head_h = seglen * 0.53

    pi = np.pi

    model_eye1 = drawEllipsoid(canvas, eye1_c, eye_l, eye_w, eye_h, eye1_br, theta + pi / 20 * rand3_eye, phi, 0)
    model_eye2 = drawEllipsoid(canvas, eye2_c, eye_l, eye_w, eye_h, eye2_br, theta - pi / 20 * rand3_eye, phi, 0)
    model_belly = drawEllipsoid(canvas, belly_c, belly_l, belly_w, belly_h, belly_br, theta, phi, 0)
    model_head = drawEllipsoid(canvas, head_c, head_l, head_w, head_h, head_br, theta, phi, 0)
    project_eye1 = project(model_eye1, dimension)
    project_eye2 = project(model_eye2, dimension)

    project_eye1 = 2 * (sigmoid(project_eye1, 0.5) - 0.7) * eyes_br
    project_eye2 = 2 * (sigmoid(project_eye2, 0.5) - 0.7) * eyes_br
    project_head = project(model_head, dimension)
    project_head = 2 * (sigmoid(project_head, 0.4) - 0.5) * head_br
    project_belly = project(model_belly, dimension)
    project_belly = 2 * (sigmoid(project_belly, 0.3) - 0.5) * belly_br
    projection = np.uint8(np.maximum(np.maximum(np.maximum(project_eye1, project_eye2), project_belly), project_head))

    return projection, eye1_c, eye2_c, model_eye1, model_eye2, model_head, model_belly
















