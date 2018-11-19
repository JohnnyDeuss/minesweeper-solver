""" Use the solver to solve a random problem, it uses the minesweeper implementation and GUI from
    https://github.com/JohnnyDeuss/minesweeper to generate minefields and display the results.
"""
from time import time, sleep
from threading import Thread
from random import randrange

from PyQt5.QtCore import pyqtSlot, QObject
import numpy as np

from minesweeper.gui import MinesweeperGUI
from solver import Solver


class Example(QObject):
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.quitting = False

    @pyqtSlot()
    def quit(self):
        self.quitting = True

    def run(self):
        gui = self.gui
        game = gui.game
        gui.aboutToQuit.connect(self.quit)
        losses = 0
        expected_losses = 0

        while not self.quitting:
            solver = Solver(game.width, game.height, game.num_mines)
            state = game.state
            while not game.done and not self.quitting:
                # Time how long a step takes to compute.
                t = time()
                prob = solver.solve(state)
                print('step - {:.5}s'.format(time() - t))
                # Flag newly found mines.
                for y, x in zip(*((prob == 1) & (state != "flag")).nonzero()):
                    gui.right_click_action(x, y)
                # Set the squares that were already opened to np.inf, so we can find the minimum of the unopened squares.
                search_mask = np.array([[isinstance(state[y][x], int) for x in range(len(state[0]))] for y in range(len(state))])
                prob[search_mask] = np.inf
                best_prob = np.nanmin(prob)
                ys, xs = (prob == best_prob).nonzero()
                if best_prob != 0:
                    # Pick a random  action from the best guesses.
                    idx = randrange(len(xs))
                    x, y = xs[idx], ys[idx]
                    print('GUESS ({:.4%}) ({}, {})'.format(best_prob, x, y))
                    expected_losses += best_prob
                    gui.left_click_action(x, y)
                else:
                    print('KNOW ({} squares)'.format(len(xs)))
                    # Open all the knowns.
                    for x, y in zip(xs, ys):
                        gui.left_click_action(x, y)
                print('L = {}, E = {}'.format(losses, expected_losses))
                sleep(1)
            if not game.is_won():
                losses += 1
            sleep(2.5)
            gui.reset()


if __name__ == '__main__':
    gui = MinesweeperGUI()
    example = Example(gui)
    example_thread = Thread(target=example.run)
    example_thread.start()
    gui.exec()
