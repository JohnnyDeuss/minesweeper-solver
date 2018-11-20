""" A policy that selects a square with minimal probability that is nearest to an already opened square.
    The distance is defined by the manhattan distance. This policy help prevent the constraint propagation from getting
    slugged down, as the main reason for this is uncertainty, allowing the solution space to grow rapidly.
"""
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from random import randrange


def nearest_policy(prob):
    # The location of the best probabilities.
    best = prob == np.nanmin(prob)
    if best.all():
        return randrange(prob.shape[1]), randrange(prob.shape[0])
    # Now keep looking further and further until a 'best' is found. We do this by dilating the mask over the known
    # squares until it overlaps with the `best` mask.
    search_mask = binary_dilation(np.isnan(prob))
    while not (search_mask & best).any():
        search_mask = binary_dilation(search_mask)
    ys, xs = (search_mask & best).nonzero()
    # Pick a random  action from the best near guesses.
    idx = randrange(len(xs))
    x, y = xs[idx], ys[idx]
    return x, y
