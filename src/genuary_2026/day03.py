"""Fibonacci Forever! Let's draw the vogel spiral.
theta = ((2*pi)/(phi**2)) * n
radius = c * root(n)
phi = golden_ratio = 0.5 * (1+root(5))
"""

from skimage.draw import disk, line, line_aa
import matplotlib.pylab as plt
import numpy as np

GRID_SIZE = 1000
MAX_N = 500
C = 1

# Calculate precursor arrays/values
rng = np.random.default_rng()
n = np.arange(MAX_N)
phi = 0.5 * (1 + np.sqrt(5))

# Generate the polar coordinates of these Vogel Spiral points
theta = ((2 * np.pi) / (phi**2)) * n
radii = C * n

# Convert the polar coords to cartesian
y, x = radii * np.sin(theta), radii * np.cos(theta)
points = np.stack([y, x]).astype(int) + (GRID_SIZE // 2)

# Draw the points to a grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

for point in points.T:
    yy, xx = disk(point, 5, shape=grid.shape)
    grid[yy, xx] = 1

for p1, p2 in rng.choice(points.T, size=(2000, 2), replace=True):
    yy, xx, vals = line_aa(*p1, *p2)
    grid[yy, xx] += vals

grid = np.log10(grid)
grid = np.clip(grid, 0, 0.2)

# Show the plots
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(grid, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day03.png", bbox_inches="tight", transparent=True, dpi=200)
plt.show()
