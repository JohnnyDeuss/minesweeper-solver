""" Note that the optimal solving procedure is more than choosing the square that's least likely square to contain a
    mine. Sometimes, even though two squares are equally likely to contain a mine, picking a certain square can be
    better, as that square's result can make subsequent guesses easier, e.g. by adding constraints to as many already
    constrained squares as possible. The exact optimal policy for picking the best square is unknown, but reinforcement
    learning can provide a way to determine a better way to pick squares than at random from the squares that are the
    least likely to contain a mine.

    Some simple policies are located in this module.
"""
from .random_policy import random_policy
from .nearest_policy import nearest_policy
