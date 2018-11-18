""" A probabilistic minesweeper solver. It determines the probability of a mine being at a certain location.
    
    The solver works in steps, first it finds all certain empty and mine squares in the boundary by using contraint
    programming. Then the probabilities for unknown squares in the boundary are calculates mathematically.
"""
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from constraint import Problem, ExactSumConstraint

from functools import reduce
from math import factorial


def solve(state, total_mines):
    """ Compute the probability of there being a mine under a given square.
        :param state: A 2D nested list representing the minesweeper state. Values can be None for an unopened square,
                      a number, 'flag' or '?'.
        :param total_mines: The total number of mines in the game (flagged or not).
    """
    # If no cells are opened, just give each cell the same probability.
    # Ignore everything but the numbers, all the rest is None.
    state = np.array([[state[y][x] if isinstance(state[y][x], int) else None for x in range(len(state[0]))] for y in range(len(state))])
    if not (state == None).all():
        solutions, boundary_mask = _find_solutions(state)
        prob = np.full(state.shape, np.inf)
        # Obviously opened cells have a 0% chance of having a mines.
        prob[state != None] = 0
        # Now mark all mines that we're certain of, because they appear in every possible solution of the boundary.
        certain_mask = boundary_mask & np.array([solutions[0] == solutions[i] for i in range(len(solutions))]).all(axis=0)
        prob[certain_mask] = solutions[0][certain_mask].astype(float)
        # Compute the number of known mines.
        # That leaves us with a couple of squares on the boundary that we're uncertain of.
        solution_mask = boundary_mask ^ certain_mask
        # Now comes the most difficult part; each solution is *not* equally likely! We need to calculate the relative
        # weight of each solution, which is proportional to the number of models that satisfy the given solution.
        N = ((state == None) ^ boundary_mask).sum()
        # Group solutions by the number of mines left unknown and outside of the solution area.
        M_known = (prob == 1).sum()
        solutions_by_M = {}
        for solution in solutions:
            # The known mines + the mines in the solutions
            M = M_known + solution[solution_mask].sum()
            M_left = total_mines - M
            # Append the solutions, making a new list if M_left isn't present yet.
            if M_left not in solutions_by_M:
                solutions_by_M[M_left] = []
            solutions_by_M[M_left].append(solution)
        # Now for each M, calculate how heaavily those solutions weigh through.
        weights = _relative_weights(solutions_by_M.keys(), N)
        # Now we just sum the weighed solutions.
        summed_weights = 0
        solution_weights = np.zeros(state.shape)
        for M, solutions in solutions_by_M.items():
            for solution in solutions:
                summed_weights += weights[M]
                solution_weights += weights[M] * solution.astype(int)
        prob[solution_mask] = solution_weights[solution_mask]/summed_weights
        # The remaining squares all have the same probability and the total probability has to equal `total_mines`.
        prob[prob == np.inf] = (total_mines - prob[prob != np.inf].sum())/N
        return prob
    else:
        return np.full(state.shape, total_mines/state.size)


def _relative_weights(Ms, N):
    """ Compute the relative weights of solutions with M mines left and N squares left. These weights are proportional
        to the number of models that have can have the given amount of mines and squares left.
    
        The number of models with N squares and M mines left is C(N, M) = N!/((N-M)!M!). To understand this formula,
        realise that the numerator is the number of permutations that of N squares. That number doesn't account for
        the permutations that have identical results. This is because two mines can be swapped without the result
        changing, the same goes for empty square. The denominator divides these duplicates out. (N-M)! divides out the
        number ways that empty squares can form duplicate results and M! divides out the the number of ways mines can
        form duplicate results. C(N, M) can be used to weigh solutions, but since C(N, M) can become very big, we can
        also compute how much more a solution weighs through compared to a solution with a different C(N, M').

        We actually don't need to know or calculate C(N, M), we just need to know how to weigh solutions relative to
        each other. To find these relative weights we look at the following equation:

        C(N, M+1) = N!/((N-(M+1)!(M+1)!)
                  = N!/(((N-M)!/(N-M+1))(M)!(M+1))
                  = N!/((N-M)!M!) * (N-M+1)/(M+1)
                  = C(N, M) * (N-M+1)/(M+1)
        Or alternatively; C(N, M) = C(N, M-1) * (N-M)/M

        So a solution with C(N, M) weighs (N-M+1)/(M+1) times less than a solution with C(N, M+1).

        :param Ms: A list of the number of mines left for which the weights will be computed.
        :param N: The number of empty squares left.
        :returns: The relative weights for each M, as a dictionary {M: weight}.
    """
    Ms = sorted(Ms)
    M = Ms[0]
    weight = 1
    weights = {}
    for M_next in Ms:
        # Iteratively compute the weights, using the above method.
        for M in range(M+1, M_next+1):
            weight *= (N-M)/(M)
        weights[M] = weight
    return weights


def _dilate(bool_ar):
    """ Perform binary dilation with a structuring element with connectivity 2. """
    return binary_dilation(bool_ar, structure=generate_binary_structure(2, 2))


def _neighbors(bool_ar):
    """ Return a binary mask marking all squares that neighbor a True cells in the boolean array """
    return bool_ar ^ _dilate(bool_ar)


def _square_mask(x, y, shape):
    """ Create a binary mask that marks only the square at (x, y). """
    mask = np.zeros(shape, dtype=bool)
    mask[y, x] = True
    return mask


def _boundary(state):
    """ Return a binary mark marking all closed squares that are adjacent to a number. """
    return _neighbors(state != None)


def _find_solutions(state):
    """ Find all possible solutions for the squares is the boundary.
        :param state: A stripped 2D nested list state, containing only Nones or numbers.
        :returns solutions: A list of solutions where a masked value that's True indicates a mine and False an empty square.
        :returns boundary_mask: The binary mask for which solutions are found, only masked values are part of the solution.
    """
    # Each square in the boundary is a variable.
    vars_mask = _boundary(state)
    ys_vars, xs_vars = vars_mask.nonzero()
    problem = Problem()
    problem.addVariables(range(len(xs_vars)), [0, 1])     # Each square can have a 0 or a 1, i.e. no mine and mine.
    # Create a 2D array to quickly look up the index of the variables given x and y coordinates
    var_lookup = np.zeros(state.shape, dtype=int)
    var_lookup[ys_vars, xs_vars] = range(len(xs_vars))
    # Find all _neighbors of the boundary that containing numbers, which define the contraints.
    constraints_mask = _neighbors(vars_mask) & (state != None)
    ys_constraints, xs_constraints = constraints_mask.nonzero()
    # For each contraint check which variables in the border they apply to and add contraints to them.
    for x, y in zip(xs_constraints, ys_constraints):
        cstrd_vars_mark = _neighbors(_square_mask(x, y, state.shape)) & vars_mask
        ys_cstrd, xs_cstrd = cstrd_vars_mark.nonzero()
        cstrd_vars = var_lookup[ys_cstrd, xs_cstrd]
        problem.addConstraint(ExactSumConstraint(state[y][x]), list(cstrd_vars))
    # Now just propagate the constraints to find squares that are certainly mines.
    solutions = problem.getSolutions()
    solution_masks = [np.zeros(state.shape, dtype=bool) for _ in range(len(solutions))]
    for solution, solution_mask in zip(solutions, solution_masks):
        ks = list(solution.keys())
        solution_mask[ys_vars[ks], xs_vars[ks]] = [solution[k] for k in ks]
    return solution_masks, vars_mask
