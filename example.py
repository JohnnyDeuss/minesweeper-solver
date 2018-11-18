""" Use the solver to solve a random problem, it uses the minesweeper implementation and GUI from
    https://github.com/JohnnyDeuss/minesweeper to generate minefields and display the results.
"""
from time import time, sleep
from threading import Thread
from random import randrange

from PyQt5.QtCore import pyqtSlot, QObject
import numpy as np

from minesweeper import Minesweeper
from minesweeper.gui import MinesweeperGUI
from solver import solve


class Solver(QObject):
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

        while True:
            state = game.state
            done = False
            while not done and not self.quitting:
                t = time()
                prob = solve(state, game.num_mines)
                print('step - {:.5}s'.format(time() - t))
                # Set the squares that were already opened to np.inf, so we can find the minimum of the unopened squares.
                search_mask = np.array([[isinstance(state[y][x], int) for x in range(len(state[0]))] for y in range(len(state))])
                prob[search_mask] = np.inf
                best_prob = prob.min()
                ys, xs = (prob == best_prob).nonzero()
                if best_prob != 0:
                    idx = randrange(len(xs))
                    x, y = xs[idx], ys[idx]
                    print('GUESS ({:.4%}) ({}, {})'.format(best_prob, x, y))
                    done, opened = game.select(x, y)
                else:
                    print('KNOW ({} squares)'.format(len(xs)))
                    opened = []
                    for x, y in zip(xs, ys):
                        done, opened_sfx = game.select(x, y)
                        opened += opened_sfx
                for square in opened:
                    gui.square_value_changed.emit(square.x, square.y, str(square.value))
                gui.move_ended.emit()
                if done:
                    print('WON' if game.is_won() else 'LOST')
                    gui.mine_counter_changed.emit(0)
                    gui.reset_value_changed.emit('won' if game.is_won() else 'lost')
                # Verify that if p was 0 the game can't have been lost.
                if best_prob == 0 and game.done and not game.is_won():
                    raise Exception('ERROR')
                sleep(1)
            sleep(5)
            game.reset()
            gui.game_reset.emit()
            gui.reset_value_changed.emit('None')
            print(game.state)


if __name__ == '__main__':
    gui = MinesweeperGUI()
    solver = Solver(gui)
    solver_thread = Thread(target=solver.run)
    solver_thread.start()
    gui.exec()
