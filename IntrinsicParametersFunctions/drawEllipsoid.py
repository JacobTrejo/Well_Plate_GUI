import numpy as np
from IntrinsicParametersFunctions.subAuxilary import rotx, roty, rotz
def drawEllipsoid(canvas, elpsd_ct, elpsd_a, elpsd_b, elpsd_c, brightness, theta, phi, gamma):
    x = np.linspace(0, canvas.shape[0] - 1, canvas.shape[0])
    y = np.linspace(0, canvas.shape[1] - 1, canvas.shape[1])
    z = np.linspace(0, canvas.shape[2] - 1, canvas.shape[2])
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='xy')
    # Co-ordinates of the ellipsoid shifted to its center
    XX = XX - elpsd_ct[0] + 1
    YY = YY - elpsd_ct[1] + 1
    ZZ = ZZ - elpsd_ct[2] + 1
    # Reorient the ellipsoid based on input angles
    rot_mat = rotx(-gamma) @ roty(-phi) @ rotz(-theta)
    XX_transformed = rot_mat[0, 0] * XX + rot_mat[0, 1] * YY + rot_mat[0, 2] * ZZ
    YY_transformed = rot_mat[1, 0] * XX + rot_mat[1, 1] * YY + rot_mat[1, 2] * ZZ
    ZZ_transformed = rot_mat[2, 0] * XX + rot_mat[2, 1] * YY + rot_mat[2, 2] * ZZ
    # Generate intensities of the model
    model = 1 + (-(XX_transformed ** 2 / (2 * elpsd_a ** 2) + YY_transformed ** 2 / (
                2 * elpsd_b ** 2) + ZZ_transformed ** 2 / (2 * elpsd_c ** 2) - 1))
    model[model < 0] = 0
    return model