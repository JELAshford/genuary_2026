"""Fibonacci Forever! Let's draw the vogel spiral.
theta = ((2*pi)/(phi**2)) * n
radius = c * root(n)
phi = golden_ratio = 0.5 * (1+root(5))
"""

import matplotlib.animation as animation
from skimage.draw import disk, line
import matplotlib.pylab as plt
from tqdm import tqdm
import numpy as np

GRID_SIZE = 1000
NUM_EDGES = 1000
MAX_N = 500
SEED = 1701
FPS = 30
FRAMES = FPS * 4
REPEAT_LEN = 10
C = 1

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xticks([])
ax.set_yticks([])

# Calculate precursor arrays/values
rng = np.random.default_rng(seed=SEED)
n = np.arange(MAX_N)
phi = 0.5 * (1 + np.sqrt(5))
pair_indexes = rng.choice(n, size=(NUM_EDGES, 2), replace=True)

# Generate the polar coordinates of these Vogel Spiral points
theta = ((2 * np.pi) / (phi**2)) * n
radii = C * n

random_changes = rng.choice(
    n, size=(FRAMES // 2 // REPEAT_LEN, int(MAX_N * 0.8)), replace=True
)
repeated_changes = np.repeat(random_changes, REPEAT_LEN, axis=0)
random_changes = np.concat([repeated_changes, repeated_changes], axis=0)

directions = np.ones(FRAMES // REPEAT_LEN // 2)
directions[::2] *= -1
directions = np.repeat(directions, REPEAT_LEN, axis=0)
directions = np.concat([directions, directions[::-1]], axis=0)

ims = []
offset = np.pi / 64
for direction, changes in tqdm(zip(directions, random_changes), total=FRAMES):
    theta[changes] = (theta[changes] + direction * offset) % (2 * np.pi)

    # Convert the polar coords to cartesian
    y, x = radii * np.sin(theta), radii * np.cos(theta)
    points = np.stack([y, x]).astype(int)
    points = (points - points.min()) / (points.max() - points.min())
    points = points * (GRID_SIZE * 0.9)
    points += (np.array([GRID_SIZE // 2, GRID_SIZE // 2]) - points.mean(axis=1))[
        :, None
    ]
    points = points.astype(int)

    # Draw the points to a grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for p1, p2 in pair_indexes:
        yy, xx = line(*points[:, p1], *points[:, p2])
        grid[yy, xx] = 0.5

    for point in points.T:
        yy, xx = disk(point, 5, shape=grid.shape)
        grid[yy, xx] = 1

    # Show the plots
    im = ax.imshow(grid, cmap="gray")
    ims.append([im])


# Create and save animation
ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=1000 / FPS,
    blit=True,
    repeat=True,
    repeat_delay=500,
)
ani.save("out/day03.gif", fps=FPS, savefig_kwargs=dict(transparent=True))
plt.close()
