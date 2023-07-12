# Minesweeper solver
A minesweeper solver that uses a number of techniques, such as constraint
programming and math, to find the exact probability that a square contains a
mine.

The solver optimally solves a single game state. To play the game optimally,
however, the solver must pick the square that is the most likely to result in
the game being won, which is not necessarily the square with the least
probability of containing a mine. This is because the opening of a square can be
more likely to constrain the values of already constrained squares, e.g. opening
an isolated square in the corner will give little information about squares in
the rest of the field. The method for picking which square to open is called the
policy and finding an optimal policy is a task for machine learning and is not
in the scope of this project. Some simple policies are given in
`/solver/policies`, which are based on some common heuristics, such as opening
the corners first.

![An example of the solver doing its thing.](/examples/example.gif)

## Installation
The solver can be installed directly from GitHub using the following command:

`pip3 install git+https://github.com/JohnnyDeuss/minesweeper-solver#egg=minesweeper_solver`

## Usage
A couple of examples of the solver being used to solve minesweeper games are
given in the `/examples` directory.

## How it works
This solver uses two approaches in sequence to calculate the exact probability
that each square contains a mine. The first approach is the very basic counting
approach, where the number of known mines next to a number is subtracted from
that number. If the reduced number is 0, then all unopened neighbors are
opened. If the reduced number is equal to the number of unopened neighbors, then
all those neighbors are flagged. This simple approach is run first because it
efficiently deals with many trivial cases, reducing the cost of running the
second approach.

The second approach is more expensive and can compute the exact probability of
all unknown squares in the boundary. The steps this approach takes are as
follows:
- Divide the unknown boundary into disconnected components.
- Divide the components into areas. Each area is a group of squares to which the
  same constraints apply, for example, two squares that are both only next to
  the same 1 and 3 are in the same area. Solving areas, rather than individual
  cells, allows for massive performance improvements in the CLP step compared to
  solving individual squares.
- Use constraint programming to find all valid models assigning a number of mines
  to each area.
- Combine model counts and probabilities of each area into aggregated counts and
  probabilities for the component.
- Combine the components and again aggregate the model counts and probabilities,
  weighing the probabilities by the model counts.
  
For more details, please reference the code and the more elaborate explanation in
the comments of `minesweeper_solver/solver.py`.
