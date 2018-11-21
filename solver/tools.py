""" Tools that are useful for working with minesweeper arrays. """
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.signal import convolve2d


def dilate(bool_ar):
    """ Perform binary dilation with a structuring element with connectivity 2. Essentially what this kind of
        dilation does is return True for any square that has a True anywhere in the 8 squares surrounding it or if
        it itself is True.
    """
    return binary_dilation(bool_ar, structure=generate_binary_structure(2, 2))


def neighbors(bool_ar):
    """ Return a binary mask marking all squares that neighbor a True cells in the boolean array. """
    return bool_ar ^ dilate(bool_ar)


def neighbors_xy(x, y, shape):
    """ Return a binary mask marking all squares that neighbor the square at (x, y). """
    return neighbors(mask_square(x, y, shape))


def mask_square(x, y, shape):
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
