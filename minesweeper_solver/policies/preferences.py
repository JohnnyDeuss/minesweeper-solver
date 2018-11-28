""" This file defines a number of preferences that can be used in `policies.make_policy`. """
import numpy as np


def no_preference(prob):
    """ This preference selects all unopened square. """
    return prob == np.nanmin(prob)


def edges(prob):
    """ This preference selects edges. """
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[0, prob.shape[0]-1], 1:prob.shape[1]-1] = True
    selection[1:prob.shape[0]-1:, [0, prob.shape[1]-1]] = True
    return selection


def corners(prob):
    """ This preference selects corners. """
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[0, 0, prob.shape[0]-1, prob.shape[0]-1], [0, prob.shape[1]-1, 0, prob.shape[1]-1]] = True
    return selection


def corners2(prob):
    """ This preference selects the squares that are offset from the corner, e.g. (1, 1) instead of (0, 0). """
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[1, 1, prob.shape[0]-2, prob.shape[0]-2], [1, prob.shape[1]-2, 1, prob.shape[1]-2]] = True
    return selection


def edges2(prob):
    """ This preference selects squares that are offset from the edge, e.g. (1, 5) instead of (0, 5). """
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[1, prob.shape[0]-2], 2:prob.shape[1]-2] = True
    selection[2:prob.shape[0]-2, [1, prob.shape[1]-2]] = True
    return selection
