import numpy as np

def sigmoid(x, scaling):
    y = 1 / (1 + np.exp(-scaling * x))
    return y

def rotx(angle):
    M = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    return M


# Rotate along y axis. Angles are accepted in radians
def roty(angle):
    M = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    return M


# Rotate along z axis. Angles are accepted in radians
def rotz(angle):
    M = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return M

def custom_round(num):
    return np.floor(num) + np.round(num - np.floor(num) + 1) - 1




