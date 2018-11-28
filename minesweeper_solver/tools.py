""" A module with common functions for working with minesweeper games. """
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.signal import convolve2d


def dilate(bool_ar):
    """ Perform binary dilation with a structuring element with connectivity 2. """
    return binary_dilation(bool_ar, structure=generate_binary_structure(2, 2))


def neighbors(bool_ar):
    """ Return a binary mask marking all squares that neighbor a True cells in the boolean array. """
    return bool_ar ^ dilate(bool_ar)


def neighbors_xy(x, y, shape):
    """ Return a binary mask marking all squares that neighbor the square at (x, y). """
    return neighbors(mask_xy(x, y, shape))


def mask_xy(x, y, shape):
    """ Create a binary mask that marks only the square at (x, y). """
    mask = np.zeros(shape, dtype=bool)
    mask[y, x] = True
    return mask


def boundary(state):
    """ Return a binary mask marking all closed squares that are adjacent to a number. """
    return neighbors(~np.isnan(state))


def count_neighbors(bool_ar):
    """ Calculate how many True's there are next to a square. """
    filter = np.ones((3, 3))
    filter[1, 1] = 0
    return convolve2d(bool_ar, filter, mode='same')

def reduce_numbers(state, mines=None):
    """ Reduce the numbers in the state to represent the number of mines next to it that have not been found yet.
        :param state: The state of the minefield.
        :param mines: The mines to use to reduce numbers
    """
    num_neighboring_mines = count_neighbors(mines)
    state[~np.isnan(state)] -= num_neighboring_mines[~np.isnan(state)]
    return state
