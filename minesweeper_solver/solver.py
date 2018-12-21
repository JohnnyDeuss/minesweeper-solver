""" A probabilistic minesweeper solver. It determines the probability of a mine being at a certain location.
    This solver is correct about all minesweeper games that uniformly distribute their mines. A minesweeper game that
    does things like trying to generate pattern with less guessing will likely have different mine distributions and
    the solver will return incorrect probabilities for uncertain squares, but will still know mines that are certain.
    An example of mines no being distributed uniformly is Windows 98's version of the 'no mine on first click' option,
    which moves the mine under the mouse.
    
    The solver works in several steps. The problem can be solved entirely with constraint programming (CP) and math, but
    since CP is expensive, we use cheap method to find known squares first, having 2 advantages:
    - The number of CP variables is reduced.
    - The boundary may become split into two pieces, which can be solved separately, improving efficiency.
    The current implementation has the following steps:
    - Reduce the numbers and solve trivial cases, reduce again and repeat until no more squares are solved.
    - Split the boundary into disconnected components that can't interact with each other.
    - For each component, split the boundary into areas that are constrained by the same constraints, which also have
      the same probability.
    - Use constraint programming to compute combinations of the number of mines that can be in the areas of each
      component.
    - Mathematically combine the components to obtain the probability. This is done by computing a number that is
      proportional to the number of models per component, merging them and dividing over a number proportional to the
      total number of mines.

    This implementation relies heavily on arrays and array operations and never iterates over the array or represents
    the minefield in a different way, e.g. computing the number of neighboring squares with mines, finding the unopened
    border or looking at neighbors is all done with array operations such dilation, convolution, element-wise logical
    operations, so understanding those is critical to understanding how some steps work. This makes the code brief and
    easy to understand if you know these concepts, without having to bother with things like bounds checking.
"""
from itertools import product
from functools import reduce
from math import factorial
from operator import mul

import numpy as np
from scipy.ndimage.measurements import label
from constraint import Problem, ExactSumConstraint, MaxSumConstraint

from .tools import *


class Solver:
    def __init__(self, width, height, total_mines, stop_on_solution=True):
        """ Initialize the solver.
            :param width: The width of the minefield.
            :param height: The height of the minefield.
            :param total_mines: The total number of mines on the minefield, flagged or not.
            :param stop_on_solution: Whether to stop as soon as a square is found that can be opened with 100%
                                     certainty. This means that the solution may have np.nan in it for squares that
                                     weren't computed.
        """
        # Known mines are 1, known empty squares are 0, uncertain squares are np.nan.
        self._total_mines = total_mines
        self.known = np.full((height, width), np.nan)
        self._stop_on_solution = stop_on_solution

    def known_mine_count(self):
        """ Returns how many mines we know the location of. """
        return (self.known == 1).sum(dtype=int)

    def mines_left(self):
        """ Returns the number of mines that we don't know the location of yet. """
        return self._total_mines - self.known_mine_count()

    def solve(self, state):
        """ Compute the probability of there being a mine under a given square.
            :param state: A 2D nested list representing the minesweeper state. Values can be a number in the range
                          [0, 8], 'flag', '?' or np.nan for any unopened squares.
            :returns: An array giving the probability that each square contains a mine. If `stop_on_solution` is set, a
                      partially computed result may be returned with a number of squares being np.nan, as they
                      weren't computed yet. Squares that have already been opened will also be np.nan.
        """
        # Convert to an easier format to solve: only the numbers remain, the rest is np.nan.
        state = np.array([[state[y][x] if isinstance(state[y][x], int) else np.nan for x in range(len(state[0]))] for y in range(len(state))])

        # Are there any opened squares; is this the first move?
        if not np.isnan(state).all():
            # Expand the known state with new information from the passed state.
            self.known[~np.isnan(state)] = 0
            # Reduce the numbers and check if there are any trivial solutions, where we have a 0 with neighbors or a
            # square with the number N in it and N unflagged neighbors.
            prob, state = self._counting_step(state)
            # Stop early if the `stop_on_solution` flag is set and we've found a safe square to open.
            if self._stop_on_solution and ~np.isnan(prob).all() and 0 in prob:
                return prob
            # Compute the probabilities for the remaining, uncertain squares.
            prob = self._cp_step(state, prob)
            return prob
        else:
            # If no cells are opened yet, just give each cell the same probability.
            return np.full(state.shape, self._total_mines/state.size)

    def _counting_step(self, state):
        """ Find all trivially easy solutions. There are 2 cases we consider:
            - A square with a 0 in it and has unflagged and unopened neighbors means that we can open all neighbors.
            - 1 square with a number that matches the number of unflagged and unopened neighbors means that we can flag
              all those neigbors.

            :param state: The unreduced state of the minefield
            :returns result: An array with known mines marked with 1, squares safe to open with 0 and everything else
                             as np.nan.
            :returns reduced_state: The reduced state, where numbers indicate the number of neighboring mines that have
                                    *not* been found.
        """
        result = np.full(state.shape, np.nan)
        # This step can be done multiple times, as each time we have results, the numbers can be further reduced.
        new_results = True
        # Subtract all numbers by the amount of neighboring mines we've already found, simplifying the game.
        state = reduce_numbers(state, self.known == 1)
        # Calculate the unknown square, i.e. that are unopened and we've not previously found their value.
        unknown_squares = np.isnan(state) & np.isnan(self.known)
        while new_results:
            num_unknown_neighbors = count_neighbors(unknown_squares)
            ### First part: squares with the number N in it and N unflagged/unopened neighbors => all mines.
            # Calculate squares with the same amount of unflagged neighbors as neighboring mines (except if N==0).
            solutions = (state == num_unknown_neighbors) & (num_unknown_neighbors > 0)
            # Create a mask for all those squares that we now know are mines. The reduce makes a neighbor mask for each
            # solution and or's them together, making one big neighbors mask.
            known_mines = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding: 1 for mines.
            self.known[known_mines] = 1
            # Further reduce the numbers, since we found new mines.
            state = reduce_numbers(state, known_mines)
            # Update what is unknown by removing known flags from the `unknown_squares` mask.
            unknown_squares = unknown_squares & ~known_mines
            # The unknown neighbor count might've changed too, so recompute it.
            num_unknown_neighbors = count_neighbors(unknown_squares)

            ### Second part: squares with a 0 in and any unflagged/unopened neighbors => all safe.
            # Calculate the squares that have a 0 in them, but still have unknown neighbors.
            solutions = (state == 0) & (num_unknown_neighbors > 0)
            # Select only those squares that are unknown and we've found to be neighboring any of the found solutions.
            # The reduce makes a neighbor mask for each solution and or's them together, making one big neighbor mask.
            known_safe = unknown_squares & reduce(np.logical_or,
                [neighbors_xy(x, y, state.shape) for y, x in zip(*solutions.nonzero())], np.zeros(state.shape, dtype=bool))
            # Update our known matrix with these new finding: 0 for safe squares.
            self.known[known_safe] = 0
            # Update what is unknown.
            unknown_squares = unknown_squares & ~known_safe

            # Now update the result matrix for both steps, 0 for safe squares, 1 for mines.
            result[known_safe] = 0
            result[known_mines] = 1
            new_results = (known_safe | known_mines).any()
        return result, state

    def _cp_step(self, state, prob):
        """ The constraint programming step.

            This is one of the more complicated steps; it divides the boundary into
            components that don't influence each other first, then divides each of those into areas that are equally
            constrained and must therefore have the same probabilities. The combinations of the number of mines in those
            components is computed with constraint programming. Those solutions are then combined to count the number of
            models in which each area has the given number of mines, from which we can calculate the average expected
            number of mines per square in a component if it has M mines, i.e. per component we have a mapping of
            {num_mines: (num_models, avg_prob)}. This information is then passed on to the combining step to form the
            final probabilities.

            :param state: The reduced state.
            :param prob: The already computed probabilities.
            :returns: The exact probability for every unknown square.
        """
        components, num_components = self._components(state)
        c_counts = []   # List of model_count_by_m instances from inside the 'for c' loop.
        c_probs = []    # List of model_count_by_m instances from inside the 'for c' loop.
        m_known = self.known_mine_count()
        # Solve each component individually
        for c in range(1, num_components+1):
            areas, constraints = self._get_areas(state, components == c)
            # Create a CP problem to determine which combination of mines per area is possible.
            problem = Problem()
            # Add all variables, each one having a domain [0, num_squares].
            for v in areas.values():
                problem.addVariable(v, range(len(v)+1))
            # Now constrain how many mines areas can have combined.
            for constraint in constraints:
                problem.addConstraint(constraint, [v for k, v in areas.items() if constraint in k])
            # Add a constraint so that the maximum number of mines never exceeds the number of mines left.
            problem.addConstraint(MaxSumConstraint(self._total_mines - m_known), list(areas.values()))
            solutions = problem.getSolutions()
            model_count_by_m = {}       # {m: #models}
            model_prob_by_m = {}        # {m: prob of the average component model}
            # Now count the number of models that exist for each number of mines in that component.
            for solution in solutions:
                m = sum(solution.values())
                # Number of models that match this solution.
                model_count = self._count_models(solution)
                # Increase counter for the number of models that have m mines.
                model_count_by_m[m] = model_count_by_m.get(m, 0) + model_count
                # Calculate the probability of each square in the component having a mines.
                model_prob = np.zeros(prob.shape)
                for area, m_area in solution.items():
                    # The area has `m_area` mines in it, evenly distributed.
                    model_prob[tuple(zip(*area))] = m_area/len(area)
                # Sum up all the models, giving the expected number of mines of all models combined
                model_prob_by_m[m] = model_prob_by_m.get(m, np.zeros(prob.shape)) + model_count*model_prob
            # We've summed the probabilities of each solution, weighted by the number of models with those
            # probabilities, now divide out the total number of models to obtain the probability of each square of a
            # model with m mines having a mine.
            model_prob_by_m = {m: model_prob/model_count_by_m[m] for m, model_prob in model_prob_by_m.items()}
            c_probs.append(model_prob_by_m)
            c_counts.append(model_count_by_m)
        prob = self._combine_components(state, prob, c_probs, c_counts)
        return prob

    def _combine_components(self, state, prob, c_probs, c_counts):
        """ Combine the probabilities and model counts found in the CP step into one probability array.

            The combining is done by forming all combinations of mine counts for each component, without exceeding the
            total number of mines in the game, and a number proportional to the total number of models that exist per
            combination. The exact probability is then the individual probabilities weighed by the number of models,
            divided by the total number of models.

            :param state: The reduced state.
            :param prob: The already computed probabilities.
            :param c_probs: A list of probability mappings per component, each having the format {num_mines: prob}
            :param c_probs: A list of model count mappings per component, each having the format {num_mines: model count}
            :returns: The exact probability for every unknown square.
        """
        # Find the unconstrained squares, for which we need to calculate the weight of models with m mines.
        solution_mask = boundary(state) & np.isnan(self.known)
        unconstrained_squares = np.isnan(state) & ~solution_mask & np.isnan(self.known)
        n = unconstrained_squares.sum(dtype=int)
        # It's possible there aren't any components at all, e.g. when an area is cut off by mines, so just skip this.
        if c_probs:
            # Instead of calculating the total number of models, we calculate weights that are proportional to the
            # number of models of the combined components, i.e. w ∝ #models. This number is a lot smaller.
            min_mines = sum([min(d) for d in c_probs])
            max_mines = sum([max(d) for d in c_probs])
            mines_left = self.mines_left()
            weights = self._relative_weights(range(min_mines, min(max_mines, mines_left)+1), n)
            # Accumulate weights and probabilities inside the upcoming for loop, where combinations of components are
            # processed.
            total_weight = 0  # The weight of solutions combined.
            total_prob = np.zeros(prob.shape)  # The summed weighted probabilities.
            # Iterate over all combinations of the components.
            for c_ms in product(*[d.keys() for d in c_probs]):
                m = sum(c_ms)
                if self.mines_left() - n <= m <= mines_left:
                    # Combine the prob arrays for this component combination.
                    comb_prob = reduce(np.add, [c_probs[c][c_m] for c, c_m in enumerate(c_ms)])
                    comb_model_count = reduce(mul, [c_counts[c][c_m] for c, c_m in enumerate(c_ms)])
                    weight = weights[m] * comb_model_count
                    # Sum up the weights and the weighted probabilities.
                    total_weight += weight
                    total_prob += weight * comb_prob
            # Now normalize the probabilities by dividing out the total weight.
            total_prob /= total_weight
            # Add result to the prob array.
            prob[solution_mask] = total_prob[solution_mask]
        # If there are any unconstrained mines...
        if n > 0:
            m_known = self.known_mine_count()
            # The amount of remaining mines is distributed evenly over the unconstrained squares.
            prob[unconstrained_squares] = (self._total_mines - m_known - prob[~np.isnan(prob) & np.isnan(self.known)].sum()) / n
        # Remember the certain values.
        certain_mask = np.isnan(self.known) & ((prob == 0) | (prob == 1))
        self.known[certain_mask] = prob[certain_mask]
        return prob

    def _count_models(self, solution):
        """ Count how many models are possible for the solution of the component areas.
            :param solution: A solution to component areas, being a dictionary {area_key: number_or_mines}.
            :returns: How many ways the component's areas can be filled to match the solution.
        """
        # Multiply the number of combinations for each individual area to get the number of models.
        return reduce(mul, [self.combinations(len(area), m) for area, m in solution.items()])

    def _components(self, state):
        """ Find all connected components in the boolean array.

            In this case, a connected component is not quite like the typical mathematical concept, as components can be
            next to each other without influencing each other, so whereas they would be part of the same component in
            the typical sense, they are not in the minesweeper sense. Furthermore, a pair of traditional components that
            are not neighbors may still be connected in minesweeper, as they could have a number in between them that
            connects the two, causing information from one traditional component to affect another component.
        """
        # Get the numbers next to unknown borders.
        numbers_mask = dilate(np.isnan(state) & np.isnan(self.known)) & ~np.isnan(state)
        labeled, num_components = label(numbers_mask)
        # Get the section of the boundary that corresponds to the previously obtained numbers.
        number_boundary_masks = [neighbors(labeled == c) & np.isnan(self.known) & np.isnan(state) for c in range(1, num_components+1)]
        # Now merge components that overlap.
        i = 0   # (For once, a C-style for loop would be nice)
        while i < len(number_boundary_masks)-1:
            j = i + 1
            while j < len(number_boundary_masks):
                # If the components overlap...
                if (number_boundary_masks[i] & number_boundary_masks[j]).any():
                    # Merge the components.
                    number_boundary_masks[i] = number_boundary_masks[i] | number_boundary_masks[j]
                    del number_boundary_masks[j]
                    # The mask that was merged in may overlap with already checked masks, so loop with the same i.
                    i -= 1  # -1, so the outer while will iterate it back to the same i after we break.
                    break
                j += 1
            i += 1
        # Number each of the resulting boundaries, as with scipy's `label`.
        labeled = np.zeros(state.shape)
        num_components = len(number_boundary_masks)
        for c, mask in enumerate(number_boundary_masks, 1):
            labeled[mask] = c
        # Now connect components that have a number in between them that connect them with a constraint.
        i = 1
        while i <= num_components-1:
            j = i + 1
            while j <= num_components:
                # If there is a number connecting the two components...
                if not np.isnan(state[dilate(labeled == i) & dilate(labeled == j)]).all():
                    # Merge the components.
                    labeled[labeled == j] = i
                    labeled[labeled > j] -= 1
                    num_components -= 1
                    # The mask that was merged in may overlap with already checked masks, so loop with the same i.
                    i -= 1  # -1, so the outer while will iterate it back to the same i after we break.
                    break
                j += 1
            i += 1
        return labeled, num_components

    @staticmethod
    def _get_areas(state, mask):
        """ Split the masked area into regions, for which each square in that region is constrained by the same
            constraints.
            :returns mapping: A mapping of constraints to an n-tuple of squares it applies to. Each of these constraint
                              tuples uniquely defines an area.
            :returns constraints: A list of all constraints applicable in the component.
        """
        # Find all squares that contain numbers that are constraints for the given state.
        constraints_mask = neighbors(mask) & ~np.isnan(state)
        # Generate a list of all CP constraints corresponding to numbers.
        constraint_list = [ExactSumConstraint(int(num)) for num in state[constraints_mask]]
        # Now create an array where those constraints are placed in the corresponding squares.
        constraints = np.full(state.shape, None, dtype=object)
        constraints[constraints_mask] = constraint_list
        # Now create an array where a list of *applicable* constraints is stored for each variable in the boundary,
        # i.e. each square aggregates the constraints defined in its neighbors.
        applied_constraints = np.empty(state.shape, dtype=object)
        for y, x in zip(*mask.nonzero()):
            applied_constraints[y, x] = []
        for yi, xi in zip(*constraints_mask.nonzero()):
            constrained_mask = neighbors_xy(xi, yi, mask.shape) & mask
            for yj, xj in zip(*constrained_mask.nonzero()):
                applied_constraints[yj, xj].append(constraints[yi, xi])
        # Now group all squares that have the same constraints applied to them, each one being an area.
        mapping = {}
        for yi, xi in zip(*mask.nonzero()):
            k = tuple(applied_constraints[yi, xi])  # Convert to tuple, so we can use it as a hash key.
            if k not in mapping:
                mapping[k] = []
            mapping[k].append((yi, xi))
        # Turn the list of (x, y) tuples into a tuple, which allows them to be used as hash keys.
        mapping = {k: tuple(v) for k, v in mapping.items()}
        return mapping, constraint_list

    @staticmethod
    def combinations(n, m):
        """ Calculate the number of ways that m mines can be distributed in n squares. """
        return factorial(n)/(factorial(n-m)*factorial(m))

    def _relative_weights(self, ms_solution, n):
        """ Compute the relative weights of solutions with M mines and N squares left. These weights are proportional
            to the number of models that have the given amount of mines and squares left.

            The number of models with N squares and M mines left is C(N, M) = N!/((N-M)!M!). To understand this formula,
            realise that the numerator is the number of permutations of N squares. That number doesn't account for the
            permutations that have identical results. This is because two mines can be swapped without the result
            changing, the same goes for empty square. The denominator divides out these duplicates. (N-M)! divides out
            the number ways that empty squares can form duplicate results and M! divides out the the number of ways
            mines can form duplicate results. C(N, M) can be used to weigh solutions, but since C(N, M) can become very
            large, we can instead compute how much more a solution weighs through compared to a solution with a
            different C(N, M').

            We actually don't need to know or calculate C(N, M), we just need to know how to weigh solutions relative to
            each other. To find these relative weights we look at the following series of equalities:

            C(N, M+1) = N! / ((N-(M+1))! (M+1)!)
                      = N! / (((N-M)!/(N-M+1)) M!(M+1))
                      = N! / ((N-M)! M!) * (N-M+1)/(M+1)
                      = C(N, M) * (N-M+1) / (M+1)
            Or alternatively: C(N, M) = C(N, M-1) * (N-M)/M
            Notice that there is however an edge case where this equation doesn't hold, where N=M+1, you'd have
            a division by zero within (N-M)!/(N-M+1).

            So, a solution with C(N, M) models weighs (N-M)/M times more than a solution with C(N, M-1) models, allowing
            us to inductively calculate relative weights.

            :param ms_solution: A list of the number of mines in the solution, from which we derive the number of mines
                                left outside of the solution, which is the M from the equations above..
            :param n: The number of empty squares left.
            :returns: The relative weights for each M, as a dictionary {M: weight}.
        """
        mines_left = self.mines_left()
        # Special case: N == 0, all solutions should have the same weight.
        if n == 0:
            return {m: 1 for m in ms_solution}
        ms = sorted(ms_solution, reverse=True)
        m = mines_left-ms[0]
        weight = 1
        weights = {}
        for m_next_solution in ms:
            m_next = mines_left - m_next_solution
            # Iteratively compute the weights, using the results computed above to update the weight.
            for m in range(m+1, m_next+1):
                # Special case: m == n, due to factorial in the derivation.
                if n == m:
                    weight *= 1/m
                else:
                    weight *= (n-m)/m
            weights[m_next_solution] = weight
        return weights
