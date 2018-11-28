""" This file defines a number of selection methods that can be used in `policies.make_policy`. """
from random import randrange

import numpy as np
from scipy.ndimage.morphology import binary_dilation


def nearest(preferred, prob):
    """ Select a square that is nearest, as defined by the Manhattan distance, to an already opened square. """
    # Nearest only works if an opened square exists to be near, otherwise just return without selecting.
    if np.isnan(prob).any():
        # Keep looking further and further until a `preferred` square is found. We do this by dilating the mask of the
        # known squares until it overlaps with the `preferred` mask.
        search_mask = binary_dilation(np.isnan(prob))
        while not (search_mask & preferred).any():
            search_mask = binary_dilation(search_mask)
        return search_mask & preferred
    else:
        return preferred


def random(preferred, prob):
    """ Select a random square. """
    ys, xs = preferred.nonzero()
    # Pick a random  action from the best guesses.
    idx = randrange(len(xs))
    x, y = xs[idx], ys[idx]
    selected = np.zeros(preferred.shape)
    selected[y, x] = 1
    return selected


def centered(preferred, prob):
    """ Select squares that are as much in the center as possible. """
    # Start at the center.
    search_mask = np.zeros(preferred.shape, dtype=bool)
    search_mask[search_mask.shape[0]//2, search_mask.shape[1]//2] = True
    # Keep looking further and further until a `preferred` square is found. We do this by dilating the mask over the
    # known squares until it overlaps with the `preferred` mask.
    while not (search_mask & preferred).any():
        search_mask = binary_dilation(search_mask)
    return search_mask & preferred


def inward(preferred, prob):
    """ Select cells that are as much on the outer edge as possible. """
    # Start at the edge.
    search_mask = np.zeros(preferred.shape, dtype=bool)
    search_mask[[0, search_mask.shape[0]-1], :] = True
    search_mask[:, [0, search_mask.shape[1]-1]] = True
    # Keep looking further and further until a `preferred` square is found. We do this by dilating the mask over the
    # known squares until it overlaps with the `preferred` mask.
    while not (search_mask & preferred).any():
        search_mask = binary_dilation(search_mask)
    return search_mask & preferred


def inward_corner(preferred, prob):
    """ Select cells that are as near to a corner as possible. """
    # Start at the center.
    search_mask = np.zeros(preferred.shape, dtype=bool)
    search_mask[[0, 0, search_mask.shape[0]-1, search_mask.shape[0]-1], [0, search_mask.shape[1]-1, 0, search_mask.shape[1]-1]] = True
    # Keep looking further and further until a `preferred` square is found. We do this by dilating the mask over the
    # known squares until it overlaps with the `preferred` mask.
    while not (search_mask & preferred).any():
        search_mask = binary_dilation(search_mask)
    return search_mask & preferred
