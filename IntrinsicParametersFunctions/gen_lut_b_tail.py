import numpy as np
from numpy.random import normal as normrnd
from scipy.stats import norm
from IntrinsicParametersFunctions.subAuxilary import custom_round


def gen_lut_b_tail(n, seglen, d1, d2, t, randomize, tail_parameters):
    ball_size_parameter = tail_parameters[0]
    ball_thickness_parameter = tail_parameters[1]
    tail_brightness = tail_parameters[2]

    size_lut = 29
    size_half = (size_lut + 1) / 2

    # Size of the balls in the ball and stick model
    random_number_size = randomize * normrnd(0.5, 0.1) + (1 - randomize) * 0.5
    ballsize = random_number_size * np.array([3, 2, 2, 2, 2, 1.5, 1.2, 1.2, 1]) * ball_size_parameter
    # Thickness of the sticks in the model
    thickness = random_number_size * np.array([7, 6, 5.5, 5, 4.5, 4, 3.5, 3]) * ball_thickness_parameter
    # Brightness of the tail
    b_tail = np.array([0.7, 0.55, 0.45, 0.40, 0.32, 0.28, 0.20, 0.15]) / 1.5 * tail_brightness

    imageSizeX = size_lut
    imageSizeY = size_lut

    columnsInImage0, rowsInImage0 = np.meshgrid(np.linspace(0, imageSizeX - 1, imageSizeX),
                                                np.linspace(0, imageSizeY - 1, imageSizeY), indexing='xy')
    imblank = np.zeros((size_lut, size_lut), dtype=np.uint8)

    radius = ballsize[n + 1]
    th = thickness[n + 1]
    bt = b_tail[n]
    bt_gradient = b_tail[n + 1] / b_tail[n]
    p_max = norm.pdf(0, 0, th)
    centerX = (size_half - 1) + d1
    centerY = (size_half - 1) + d2
    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0
    ballpix = (rowsInImage - centerY) ** 2 + (columnsInImage - centerX) ** 2 <= radius ** 2
    ballpix = custom_round(custom_round(np.uint8(ballpix) * 255 * bt) * 0.85)
    pt = np.zeros((2, 2))
    R = np.squeeze(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))
    vec = np.matmul(R, np.array([seglen, 0], dtype=np.float64))
    pt[:, 0] = np.array([(size_half - 1) + d1, (size_half - 1) + d2])
    pt[:, 1] = pt[:, 0] + vec
    stickpix = imblank
    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0
    if (pt[0, 1] - pt[0, 0]) != 0:
        slope = (pt[1, 1] - pt[1, 0]) / (pt[0, 1] - pt[0, 0])
        # vectors perpendicular to the line segment
        # th is the thickness of the sticks in the model
        vp = np.array([-slope, 1]) / np.linalg.norm(np.array([-slope, 1]))
        # one vertex of the rectangle
        V1 = pt[:, 1] - vp * th
        # two sides of the rectangle
        s1 = 2 * vp * th
        s2 = pt[:, 0] - pt[:, 1]
        # find the pixels inside the rectangle
        r1 = rowsInImage - V1[1]
        c1 = columnsInImage - V1[0]
        # innter products
        ip1 = r1 * s1[1] + c1 * s1[0]
        ip2 = r1 * s2[1] + c1 * s2[0]
        condition1_mask = np.zeros((ip1.shape[0], ip1.shape[1]), dtype=bool)
        condition1_mask[ip1 > 0] = True
        condition2_mask = np.zeros((ip1.shape[0], ip1.shape[1]), dtype=bool)
        condition2_mask[ip1 < np.dot(s1, s1)] = True
        condition3_mask = np.zeros((ip2.shape[0], ip2.shape[1]), dtype=bool)
        condition3_mask[ip2 > 0] = True
        condition4_mask = np.zeros((ip2.shape[0], ip2.shape[1]), dtype=bool)
        condition4_mask[ip2 < np.dot(s2, s2)] = True
        stickpix_bw = np.logical_and.reduce((condition1_mask, condition2_mask, condition3_mask, condition4_mask))
    else:
        condition1_mask = np.zeros(rowsInImage.shape[0], rowsInImage.shape[1], dtype=bool)
        condition1_mask[rowsInImage < np.maximum(pt[1, 1], pt[1, 0])] = True
        condition2_mask = np.zeros(rowsInImage.shape[0], rowsInImage.shape[1], dtype=bool)
        condition2_mask[rowsInImage > np.minimum(pt[1, 1], pt[1, 0])] = True
        condition3_mask = np.zeros(columnsInImage.shape[0], columnsInImage.shape[1], dtype=bool)
        condition3_mask[columnsInImage < pt[0, 1] + th] = True
        condition4_mask = np.zeros(columnsInImage.shape[0], columnsInImage.shape[1], dtype=bool)
        condition4_mask[columnsInImage > pt[0, 1] - th] = True
        stickpix_bw = np.logical_and.reduce(condition1_mask, condition2_mask, condition3_mask, condition4_mask)

    # brightness of the points on the stick is a function of its distance to the segment
    idx_bw = np.argwhere(stickpix_bw == 1)
    ys = idx_bw[:, 0]
    xs = idx_bw[:, 1]
    px = pt[0, 1] - pt[0, 0]
    py = pt[1, 1] - pt[1, 0]
    pp = px * px + py * py
    # the distance between a pixel and the fish backbone
    d_radial = np.zeros((len(ys), 1))
    # the distance between a pixel and the anterior end of the segment (0 < d_axial < 1)
    b_axial = np.zeros((len(ys), 1))
    for i in range(0, len(ys)):
        u = ((xs[i] - pt[0, 0]) * px + (ys[i] - pt[1, 0]) * py) / pp
        dx = pt[0, 0] + u * px - xs[i]
        dy = pt[1, 0] + u * py - ys[i]
        d_radial[i] = dx * dx + dy * dy
        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9
    b_stick = np.uint8(255 * (norm.pdf(d_radial, 0, th) / p_max))
    for i in range(0, len(ys)):
        stickpix[ys[i], xs[i]] = custom_round(b_stick[i] * b_axial[i])
    stickpix = custom_round(stickpix * bt)
    graymodel = np.maximum(ballpix, stickpix)

    return graymodel






