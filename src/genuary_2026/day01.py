"""One colour, one shape. Let's make like a tree!"""

from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
from skimage.draw import disk
import numpy as np


GRID_SIZE = 500
NUM_STEPS = 90
NUM_POINTS = 300
SEED = 1701
STEP_SIZE = 10
MAX_RADIUS = 20

# Setup RNG
rng = np.random.default_rng(seed=SEED)

# define random targets
targets = rng.random(size=(2, NUM_POINTS))
targets[0, :] = ((targets[0, :] * 1.4) / 2) * GRID_SIZE
targets[1, :] = ((((targets[1, :] * 1.5) / 2) + 0.65) % 1) * GRID_SIZE

targets = targets.astype(int)

# define starting node
node_ages = [0]
nodes = np.array([[GRID_SIZE], [GRID_SIZE // 2]])

# Run the algorithm, creating new nodes to move towards targets
for val in range(NUM_STEPS):
    distance_thresh = ((NUM_STEPS - val) / NUM_STEPS) * GRID_SIZE
    node_target_distances = cdist(nodes.T, targets.T)
    node_target_distances[node_target_distances > distance_thresh] = np.inf
    closest_node_distances = np.min(node_target_distances, axis=0)
    closest_nodes_to_targets = np.argmin(node_target_distances, axis=0)
    valid_nodes = set(
        [
            node
            for node, dist in zip(closest_nodes_to_targets, closest_node_distances)
            if dist != np.inf
        ]
    )
    for node_index in valid_nodes:
        node_position = nodes[:, node_index]
        target_positions = targets[
            :, np.where(closest_nodes_to_targets == node_index)
        ].squeeze()
        average_direction_to_targets = np.mean(
            target_positions - node_position[:, None], axis=1
        )
        magnitude = np.linalg.norm(average_direction_to_targets) + 1e-6
        average_direction_to_targets += rng.random(size=(2,)) * magnitude * 0.5
        direction_norm = average_direction_to_targets / magnitude
        direction_step = (direction_norm * STEP_SIZE).astype(int).T
        new_point = node_position + direction_step
        nodes = np.concat([nodes, new_point[:, None]], axis=-1)
        node_ages.append(val)


# Create grid to draw on, and add
grid = np.zeros((GRID_SIZE, GRID_SIZE, 3))


# Draw the nodes
for (y, x), age in zip(nodes.T, node_ages):
    radius = MAX_RADIUS * ((NUM_STEPS - age) / NUM_STEPS) ** 1.5
    rr, cc = disk((y, x), radius, shape=grid.shape)
    grid[rr, cc, 1] = 1

# Draw the random targets
for y, x in targets.T:
    rr, cc = disk((y, x), 2, shape=grid.shape)
    grid[rr, cc, 2] = 1

# Show and save grid
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day01.png", bbox_inches="tight", transparent=True)
plt.show()
