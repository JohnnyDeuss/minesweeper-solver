""" Use the solver to solve a random problem, it uses the minesweeper implementation and GUI from
    https://github.com/JohnnyDeuss/minesweeper to generate minefields and display the results.
"""
from time import sleep
from threading import Thread

from PyQt5.QtCore import pyqtSlot, QObject
import numpy as np

from minesweeper.gui import MinesweeperGUI
from solver import Solver
from solver.policies import corner_then_edge2_policy


class Example(QObject):
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.quitting = False

    @pyqtSlot()
    def quit(self):
        """ Receives exit messages from the GUI. """
        self.quitting = True

    def run(self):
        gui = self.gui
        game = gui.game
        # Uncomment to make the probabilities come out right.
        #game.set_config(first_never_mine=False)    # Disable to make first move probability is the same as the solver's?
        gui.aboutToQuit.connect(self.quit)
        wins = 0
        games = 0
        expected_wins = 0

        while not self.quitting:
            expected_win = 1
            games += 1
            solver = Solver(game.width, game.height, game.num_mines)
            state = game.state
            while not game.done and not self.quitting:
                prob = solver.solve(state)
                # Flag newly found mines.
                for y, x in zip(*((prob == 1) & (state != "flag")).nonzero()):
                    gui.right_click_action(x, y)
                best_prob = np.nanmin(prob)
                ys, xs = (prob == best_prob).nonzero()
                if best_prob != 0:
                    expected_win *= (1-best_prob)
                    x, y = corner_then_edge2_policy(prob)
                    print('GUESS ({:.4%}) ({}, {})'.format(best_prob, x, y))
                    gui.left_click_action(x, y)
                else:
                    # Open all the knowns.
                    for x, y in zip(xs, ys):
                        gui.left_click_action(x, y)
                sleep(0.01)
            expected_wins += expected_win
            if game.is_won():
                wins += 1
            print('{} | E[GameWin%] = {:.3}, Wins = {}, E[Wins] = {:.3}, Win% = {:.3%}'.format(
                'W' if game.is_won() else 'L', expected_win, wins, expected_wins, wins/games))
            sleep(5)
            gui.reset()


if __name__ == '__main__':
    gui = MinesweeperGUI(debug_mode=True, difficulty='intermediate')
    example = Example(gui)
    example_thread = Thread(target=example.run)
    example_thread.start()
    gui.exec()
