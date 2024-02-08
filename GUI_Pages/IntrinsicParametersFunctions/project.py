import numpy as np

def project(model, dimension):
    vec = np.array([0, 1, 2])
    idx = np.argwhere(vec == dimension)
    mask = np.ones(len(vec), dtype=bool)
    mask[idx] = False
    vec = vec[mask]

    if dimension == 0:
        projection = np.squeeze(np.sum(model, axis=dimension))
        projection = projection.T
        projection = np.flip(projection, axis=0)

    elif dimension == 1:
        projection = np.squeeze(np.sum(model, axis=dimension))
        projection = np.flip(projection.T, axis=1)
        projection = np.flip(projection, axis=0)

    elif dimension == 2:
        projection = np.squeeze(np.sum(model, axis=dimension))
    return projection


