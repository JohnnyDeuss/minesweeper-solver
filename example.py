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

        while True:
            solver = Solver(game.width, game.height, game.num_mines)
            state = game.state
            done = False
            while not done and not self.quitting:
                # Time how long a step takes to compute.
                t = time()
                prob = solver.solve(state)
                print('step - {:.5}s'.format(time() - t))
                # Flag newly found mines.
                for y, x in zip(*((prob == 1) & (state != "flag")).nonzero()):
                    game.flag(x, y)
                    gui.square_value_changed.emit(x, y, "flag")
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
                    done, opened = game.select(x, y)
                else:
                    print('KNOW ({} squares)'.format(len(xs)))
                    opened = []
                    # Open all the knowns.
                    for x, y in zip(xs, ys):
                        done, opened_sfx = game.select(x, y)
                        opened += opened_sfx
                for square in opened:
                    gui.square_value_changed.emit(square.x, square.y, str(square.value))
                gui.move_ended.emit()
                if done:
                    if game.is_won():
                        print('WON')
                    else:
                        print('LOST')
                        losses += 1
                    gui.mine_counter_changed.emit(0)
                    gui.reset_value_changed.emit('won' if game.is_won() else 'lost')
                    print('L = {}, E = {}'.format(losses, expected_losses))
                # Verify that if p was 0 the game can't have been lost.
                if best_prob == 0 and game.done and not game.is_won():
                    raise Exception('ERROR')
                sleep(1)
            sleep(3.5)
            game.reset()
            gui.game_reset.emit()
            gui.reset_value_changed.emit('None')


if __name__ == '__main__':
    gui = MinesweeperGUI()
    example = Example(gui)
    example_thread = Thread(target=example.run)
    example_thread.start()
    gui.exec()
