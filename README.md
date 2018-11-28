# Minesweeper solver
A minesweeper solver that uses a number of techniques, such as constraint programming and math, to find the exact
probability that squares contain a mine. It optimally solves a single state, but picking the right square relies on more
than knowing the exact probabilities, as each move can make subsequent moves easier by, for example, being more likely
to constrain the value of already constrained squares. The method for picking which square to open is called the policy
and finding a near-optimal policy requires machine learning. Some simple policies are given in `/solver/policies`.

![An example of the solver doing its thing.](/examples/example.gif)

## Installation
The solver can be installed directly from GitHub
using the following command:

`pip3 install git+https://github.com/JohnnyDeuss/minesweeper-solver#egg=minesweeper_solver`

## Usage
A couple of examples of the solver being used to solve minesweeper games are given in the `/examples` directory.

## How it works
This solver uses two approaches in sequence to solve the problem. The first approach is the very basic counting
approach, where the number of known mines next to a number is subtracted from that number. If the reduced number is 0,
then all unopened neighbors are opened. If the reduced number is equal to the number of unopened neighbors, then all
those neighbors are flagged. This simple approach is run first because it efficiently deals with trivial cases,
reducing the cost of running the second approach, not only by solving trivial squares but also by potentially breaking
up the boundary into disconnected components.

The second approach is more expensive and can computes the exact probability of all unknown squares in the boundary.
The steps this approach takes are as follows:
- Divide the unknown boundary into disconnected components.
- Divide the components into areas, each area is a group of squares to which the same constraints apply, i.e. two
  squares that are both next to only the same 1 and 3. 
- Use constraint programming to find all valid models assigning a number of mines to each area.
- Combine model counts and probabilities of each area into aggregated counts and probabilities for the component.
- Combine the components and again aggregate the model counts and probabilities, then weigh the probabilities by the
  model counts.
  
For more detail, please reference the code and the more elaborate explanation in the comments of
`minesweeper_solver/solver.py`.
