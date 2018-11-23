""" A probabilistic minesweeper solver. It determines the probability of a mine being at a certain location.
    This solver is correct about all minesweeper games that uniformly distributes its mines. A minesweeper game that
    does things like trying to generate pattern with less guessing will likely have different mine distributions and
    will return incorrect probabilities for uncertain squares. Another example of uniform mines distribution is Windows
    98's version of the no mine on first click implementation, which moves the mine under the mouse.
    
    The solver works in several steps. The problem can be solved entirely with CP and math, but since CP is expensive,
    we use cheap method for finding known squares first, having 2 advantages:
    - The number of variables is CP is reduced.
    - The boundary could be split up into two pieces, which can be solved separately, greatly improving efficiency.
    The current implementation has the following three steps:
    - Reduce the numbers and find solve trivial cases, reduce again and repeat until no more squares are solved.
    - Use constraint programming to find solutions to the remaining boundary squares.
    - Mathematically combine the solutions to get the probability.

    This implementation relies heavily on arrays and array operations and never iterates over the array or represents
    the minefield in a different way, e.g. computing the number of neighboring squares with mines, finding the unopened
    border or looking at neighbors is all done with array operations such dilation, convolution, element-wise logical
    operations, so understanding those is critical to understanding how some steps work. This makes the code brief and
    easy to understand if you know these concepts, without having to bother with things like bounds checking.
"""
from functools import reduce

import numpy as np
from constraint import Problem, ExactSumConstraint, MaxSumConstraint

from .tools import *


class Solver:
    def __init__(self, width, height, total_mines, stop_on_solution=True):
        """ Initialize the solver.
            :param width: The width of the minefield.
            :param height: The height of the minefield.
            :param total_mines: The total number of mines on the minefield, flagged or not.
            :param stop_on_solution: Whether to stop as soon as a square is found that can be opened with 100% certainty.
                                     This means that the solution may have np.nan in it for squares that weren't computed.
            :param total_mines: The total number of mines in the game (flagged or not).
        """
        # Known mines are 1, known empty squares are 0, uncertain are np.nan.
        self._total_mines = total_mines
        self._known = np.full((height, width), np.nan)
        self._stop_on_solution = stop_on_solution

    def solve(self, state):
        """ Compute the probability of there being a mine under a given square.
            :param state: A 2D nested list representing the minesweeper state. Values can be np.nan for an unopened square,
                          a number, 'flag' or '?'.
            :returns: An array giving the probability that each square contains a mine. If `stop_on_solution` is set, a
                      partially computed result may be returned with a number of squares being np.nan, as they
                      weren't computed yet. Squares that have already been opened will also be np.nan.
        """
        # Convert to an easier format to solve: only the numbers remain, the rest is np.nan.
        state = np.array([[state[y][x] if isinstance(state[y][x], int) else np.nan for x in range(len(state[0]))] for y in range(len(state))])

        # If there are no opened squares yet, everything is equally likely.
        if not np.isnan(state).all():
            # Expand the known state with new information from the passed state.
            self._known[~np.isnan(state)] = 0     # All squares that are not np.nan are opened and known.
            # Reduce the numbers and check if there are any trivial solutions, where we have a 0 with neighbors or
            # square with the number N with N unflagged neighbors.
            prob, state = self._counting_step(state)
            # Stop early if the early stopping flag is set and we've found a safe square to open?
            if self._stop_on_solution and ~np.isnan(prob).all() and 0 in prob:
                return prob
            # Compute all possible solutions of the boundary.
            solutions = self._cp_step(state)
            # There may not be solutions if the boundary doesn't contain *any* unsolved squares.
            if solutions:
                solution_mask = ~np.isnan(solutions[0])
                # Now mark known squares, because they appear in every possible solution of the boundary.
                certain_mask = solution_mask & np.array([solutions[0] == solutions[i] for i in range(len(solutions))]).all(axis=0)
                prob[certain_mask] = solutions[0][certain_mask]
                # Also update the known array that we're keeping with these values.
                self._known[certain_mask] = solutions[0][certain_mask].astype(float)
                # Simplify the solutions by dropping the certain squares from all solutions.
                solutions = [np.where(certain_mask, np.nan, solution) for solution in solutions]
                # Stop early if the early stopping flag is set and we've found a safe square to open?
                if self._stop_on_solution and ~np.isnan(prob).all() and 0 in prob:
                    return prob
            # Now combine the solutions into one probability.
            prob = self._combining_step(state, prob, solutions)
            # It's still possible to find certain values if they appear outside of the boundary, so remember them.
            certain_mask = ((prob == 0) | (prob == 1)) & np.isnan(self._known)
            self._known[certain_mask] = prob[certain_mask]
            return prob
        else:
            # If no cells are opened, just give each cell the same probability.
            return np.full(state.shape, self._total_mines/state.size)

    def _counting_step(self, state):
        """ Find all trivially easy solutions, i.e. a square with a 0 in it that has unflagged and unopened neighbors
            (= open all those neighbors) or a square with a number that matched the number of unflagged and unopened
            neighbors (= flag all those neighbors).
            :param state: The unreduced state of the minefield
            :returns result: An array with known mines marked with 1, squares safe to open with 0 and everything else as np.nan.
            :returns reduced_state: The reduced state, where numbers indicate the number of mines that are *not* found.
        """
        result = np.full(state.shape, np.nan)
        # This step can be done multiple times, as each time we have results, more numbers can be reduced.
        new_results = True
        # Subtract all numbers by the amount of neighboring mines we've already found, simplifying the game.
        state = reduce_numbers(state, self._known == 1)
        # Calculate the unknown square, i.e. unopened and we've not previously found their value (is in `self._known`).
        unknown_squares = np.isnan(state) & np.isnan(self._known)
        while new_results:
            num_unknown_neighbors = count_neighbors(unknown_squares)
            ### Second part: finding squares with number N and N unflagged/unopened neighbors (which must all be mines).
            # Calculate squares with the same amount of unflagged neighbors as neighboring mines.
            solutions = (state == num_unknown_neighbors) & (num_unknown_neighbors > 0)
            # Again, create a mask for all those squares that we now know are mines.
            # The reduce makes a neighbor mask for each solution and or's them together, making one big neighbor mask.
            known_mines = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding; 1 for mines.
            self._known[known_mines] = 1
            # Further reduce the numbers first.
            state = reduce_numbers(state, known_mines)
            # Update what is unknown.
            unknown_squares = unknown_squares & ~known_mines
            ### First part: finding squares with a 0 in and unflagged/unopened neighbors (which must all be safe).
            # Calculate the square that have a 0 on them, but still have unknown neighbors.
            solutions = (state == 0) & (num_unknown_neighbors > 0)
            # Select only those squares that are unknown and we've found to be neighboring any of the found solutions.
            # The reduce makes a neighbor mask for each solution and or's them together, making one big neighbor mask.
            known_safe = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding; 0 for safe squares.
            self._known[known_safe] = 0
            # Update what is unknown.
            unknown_squares = unknown_squares & ~known_safe
            # Now update the result matrix, 0 for safe squares, 1 for mines.
            result[known_safe] = 0
            result[known_mines] = 1
            new_results = (known_safe | known_mines).any()
        return result, state

    def _cp_step(self, state):
        """ Constraint programming step. Find all possible solutions for the squares is the boundary.
            :param state: The reduced state of the minefield.
            :returns solutions: A list of solutions where a masked value that's True indicates a mine and False an empty square.
        """
        # Each square in the boundary is a variable, except where we already know the value.
        vars_mask = boundary(state) & np.isnan(self._known)
        # Start a CP problem to work with (= CP solver).
        problem = Problem()
        variable_names = range(np.count_nonzero(vars_mask))
        problem.addVariables(variable_names, [False, True])
        # Create a 2D array to quickly look up the `Problem` name of the variable squares.
        var_lookup = np.zeros(state.shape, dtype=int)
        var_lookup[vars_mask.nonzero()] = variable_names
        # Find all _neighbors of the variables that containing numbers, which define their constraints.
        constraints_mask = neighbors(vars_mask) & ~np.isnan(state)
        # For each constraint check which variables they apply to and add constraints to them.
        for y, x in zip(*constraints_mask.nonzero()):
            constrained_vars_mask = neighbors_xy(x, y, vars_mask.shape) & vars_mask
            constrained_var_names = var_lookup[constrained_vars_mask.nonzero()]
            problem.addConstraint(ExactSumConstraint(state[y][x]), list(constrained_var_names))
        # Add a constraint to the total number of mines.
        problem.addConstraint(MaxSumConstraint(self._total_mines - (self._known == 1).sum(dtype=int)), variable_names)
        # Now just propagate the constraints to find squares that are certainly mines.
        solutions = problem.getSolutions()
        solution_masks = [np.full(state.shape, np.nan) for _ in range(len(solutions))]
        ys_vars, xs_vars = vars_mask.nonzero()  # Used for reverse lookup var_name -> (x, y)
        for solution, solution_mask in zip(solutions, solution_masks):
            ks = list(solution.keys())
            solution_mask[ys_vars[ks], xs_vars[ks]] = [solution[k] for k in ks]
        return solution_masks

    def _combining_step(self, state, prob, solutions):
        """ Combine the solutions into probabilities.
            :param state: The state of the minesweeper game.
            :param prob: The current probability array, which at this point should contain only 1's, 0's and np.nan's.
            :param solutions: A list of solution arrays from the CP step, with certain values removed, as they're
                              already in `self._known`.
        """
        # Get the  squares on the boundary that we're uncertain of.
        solution_mask = ~np.isnan(solutions[0]) if solutions else np.zeros(state.shape, dtype=bool)
        # Now comes the most difficult part; each solution is *not* equally likely! We need to calculate the
        # relative weight of each solution, which is proportional to the number of models that satisfy the given
        # solution.
        unconstrained_squares = np.isnan(state) & ~solution_mask & np.isnan(self._known)
        n = unconstrained_squares.sum(dtype=int)
        # In some rare cases, there are no unsolved squares in the boundary and we don't need to do CP.
        if solution_mask.any():
            # Group solutions by the number of mines left unknown and outside of the solution area.
            m_known = (self._known == 1).sum(dtype=int)
            solutions_by_m = {}
            for solution in solutions:
                # The known mines + the mines in the solutions
                m = m_known + solution[solution_mask].sum(dtype=int)
                m_left = self._total_mines - m
                # Append the solutions, making a new list if M_left isn't present yet.
                if m_left not in solutions_by_m:
                    solutions_by_m[m_left] = []
                solutions_by_m[m_left].append(solution)
            # Now for each M, calculate how heavily those solutions weigh through.
            weights = self._relative_weights(solutions_by_m.keys(), n)
            # Now we just sum the weighed solutions.
            summed_weights = 0
            summed_solution = np.zeros(state.shape)
            for m, solutions in solutions_by_m.items():
                for solution in solutions:
                    summed_weights += weights[m]
                    summed_solution += weights[m] * solution.astype(int)
            prob[solution_mask] = summed_solution[solution_mask] / summed_weights
        # The remaining squares all have the same probability and the total probability has to equal `total_mines`.
        if n > 0:
            # Recompute m_known in case we haven't yet.
            m_known = m_known if 'm_known' in locals() else (self._known == 1).sum(dtype=int)
            # The amount of remaining mines is distributed evenly over the unconstrained squares.
            prob[unconstrained_squares] = (self._total_mines - m_known - prob[~np.isnan(prob) & (prob != 1)].sum()) / n
        return prob

    @staticmethod
    def _relative_weights(ms, n):
        """ Compute the relative weights of solutions with M mines and N squares left. These weights are proportional
            to the number of models that have can have the given amount of mines and squares left.

            The number of models with N squares and M mines left is C(N, M) = N!/((N-M)!M!). To understand this formula,
            realise that the numerator is the number of permutations that of N squares. That number doesn't account for
            the permutations that have identical results. This is because two mines can be swapped without the result
            changing, the same goes for empty square. The denominator divides these duplicates out. (N-M)! divides out
            the number ways that empty squares can form duplicate results and M! divides out the the number of ways
            mines can form duplicate results. C(N, M) can be used to weigh solutions, but since C(N, M) can become very
            big, we can also compute how much more a solution weighs through compared to a solution with a different
            C(N, M').

            We actually don't need to know or calculate C(N, M), we just need to know how to weigh solutions relative to
            each other. To find these relative weights we look at the following series of equalities equalities:

            C(N, M+1) = N! / ((N-(M+1))! (M+1)!)
                      = N! / (((N-M)!/(N-M+1)) M!(M+1))
                      = N! / ((N-M)! M!) * (N-M+1)/(M+1)
                      = C(N, M) * (N-M+1) / (M+1)
            Or alternatively: C(N, M) = C(N, M-1) * (N-M)/M
            Notice that there is however an edge case where this equation doesn't hold, where N=M+1, you'd have
            a division by zero within (((N-M)!/(N-M+1)).

            So a solution with C(N, M) models weighs (N-M)/M times more than a solution with C(N, M-1) models, allowing
            us to inductively calculate relative weights.

            :param Ms: A list of the number of mines left for which the weights will be computed.
            :param N: The number of empty squares left.
            :returns: The relative weights for each M, as a dictionary {M: weight}.
        """
        # Special case: N == 0, all solutions should have the same weight.
        if n == 0:
            return {m: 1 for m in ms}
        ms = sorted(ms)
        m = ms[0]
        weight = 1
        weights = {}
        for m_next in ms:
            # Iteratively compute the weights, using the results computed above to update the weight.
            for m in range(m+1, m_next+1):
                # Special case: m == n, due to factorial in the derivation.
                if n == m:
                    weight *= 1/m
                else:
                    weight *= (n-m)/m
            weights[m] = weight
        return weights
