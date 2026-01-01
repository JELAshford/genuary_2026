"""One colour, one shape. Let's make like a tree!"""

from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
from skimage.draw import disk
import numpy as np


GRID_SIZE = 500
NUM_POINTS = 300
SEED = 1702
DISTANCE_THRESH = GRID_SIZE // 5
STEP_SIZE = 10
MAX_RADIUS = 10

# Setup RNG
rng = np.random.default_rng(seed=SEED)

# define random targets
targets = rng.random(size=(2, NUM_POINTS))
targets[0, :] = ((targets[0, :] * 1.5) / 2) * GRID_SIZE
targets[1, :] *= GRID_SIZE
targets = targets.astype(int)

# define starting node
# nodes = np.array(
#     [
#         [GRID_SIZE, GRID_SIZE, GRID_SIZE],
#         [GRID_SIZE // 2, (GRID_SIZE // 2) - 2, (GRID_SIZE // 2) + 2],
#     ]
# )
nodes = np.array([[GRID_SIZE], [GRID_SIZE // 2]])


# Run the algorithm, creating new nodes to move towards targets
for val in range(100):
    print(nodes.shape)
    node_target_distances = cdist(nodes.T, targets.T)
    closest_nodes_to_targets = np.argmin(node_target_distances, axis=0)
    for node_index in set(closest_nodes_to_targets):
        node_position = nodes[:, node_index]
        target_positions = targets[
            :, np.where(closest_nodes_to_targets == node_index)
        ].squeeze()
        average_direction_to_targets = np.mean(
            target_positions - node_position[:, None], axis=1
        )
        direction_norm = average_direction_to_targets / np.linalg.norm(
            average_direction_to_targets
        )
        direction_step = (direction_norm * STEP_SIZE).astype(int).T
        new_point = node_position + direction_step
        nodes = np.concat([nodes, new_point[:, None]], axis=-1)


# Create grid to draw on, and add
grid = np.zeros((GRID_SIZE, GRID_SIZE, 3))

# Draw the random targets
for y, x in targets.T:
    rr, cc = disk((y, x), 2, shape=grid.shape)
    grid[rr, cc, 2] = 1

# Draw the nodes
for y, x in nodes.T:
    radius = MAX_RADIUS * (y / GRID_SIZE) ** 0.5
    rr, cc = disk((y, x), radius, shape=grid.shape)
    grid[rr, cc, 1] = 1

# Show and save grid
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day01.png", bbox_inches="tight", transparent=True)
plt.show()
