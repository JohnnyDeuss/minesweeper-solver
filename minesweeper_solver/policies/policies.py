""" Some predefined policies made with make_policy. """
from .make_policy import make_policy
from .preferences import *
from .selection_methods import *

# Random.
random_policy = make_policy(no_preference, random)
# Nearest to already opened and preferring to work towards the middle.
nearest_policy = make_policy(no_preference, [nearest, centered])
# Prefer the corners and nearest to opened squares more towards the corner.
corner_policy = make_policy(corners, [nearest, inward_corner])
# Prefer edges and nearest opened squares more towards the edge.
edge_policy = make_policy(edges, [nearest, inward])
# Prefer corners, then edges and nearest opened squares more towards the corner.
corner_then_edge_policy = make_policy([corners2, edges2], [nearest, inward_corner])
# Same as `corner_policy`, except the corner is offset from the sides.
corner2_policy = make_policy(corners2, [nearest, inward_corner])
# Same as `edge_policy`, except the edge is offset from the sides.
edge2_policy = make_policy(edges2, [nearest, inward])
# Same as `corner_then_edge_policy`, except the corner and edge is offset from the sides.
corner_then_edge2_policy = make_policy([corners2, edges2], [nearest, inward_corner])
