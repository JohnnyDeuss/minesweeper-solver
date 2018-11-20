""" This policy randomly selects one of the squares that is least likely to contain a mine. """
import numpy as np

from random import randrange


def random_policy(prob):
    best_prob = np.nanmin(prob)
    ys, xs = (prob == best_prob).nonzero()
    # Pick a random  action from the best guesses.
    idx = randrange(len(xs))
    x, y = xs[idx], ys[idx]
    return x, y
