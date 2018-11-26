""" Use the solver to solve a random problem, it uses the minesweeper implementation from
    https://github.com/JohnnyDeuss/minesweeper to generate minefields.
"""
import numpy as np

from minesweeper import Minesweeper
from minesweeper_solver import Solver
from minesweeper_solver.policies import corner_then_edge2_policy


game = Minesweeper(difficulty='intermediate')
# Uncomment to make the probabilities come out right.
#game.set_config(first_never_mine=False)    # Disable to make first move probability is the same as the solver's?
wins = 0
games = 0
expected_wins = 0

for i in range(100000):
    expected_win = 1
    games += 1
    solver = Solver(game.width, game.height, game.num_mines)
    state = game.state
    while not game.done:
        prob = solver.solve(state)
        # Flag newly found mines.
        for y, x in zip(*((prob == 1) & (state != "flag")).nonzero()):
            game.flag(x, y)
        best_prob = np.nanmin(prob)
        ys, xs = (prob == best_prob).nonzero()
        if best_prob != 0:
            expected_win *= (1-best_prob)
            x, y = corner_then_edge2_policy(prob)
            game.select(x, y)
        else:
            # Open all the knowns.
            for x, y in zip(xs, ys):
                game.select(x, y)
    expected_wins += expected_win
    if game.is_won():
        wins += 1
    print('{}> {} | E[GameWin%] = {:.3}, Wins = {}, E[Wins] = {:.3}, Win% = {:.3%}'.format(
        i, 'W' if game.is_won() else 'L', expected_win, wins, expected_wins, wins/games))
    game.reset()
