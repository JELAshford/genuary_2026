"""Twelve principles of animation! Ease and squish"""

import matplotlib.animation as animation
from skimage.draw import ellipse
import matplotlib.pylab as plt
from autograd import jacobian
import autograd.numpy as np


def ease_inoutsine(t: np.ndarray):
    return -(np.cos(np.pi * t) - 1) / 2


def lerp(from_point: np.ndarray, to_point: np.ndarray, ts: np.ndarray):
    """{from,to}_point must be in (dim, 1) layout (e.g. for 2d points its (2, 1))."""
    return (1 - ts) * from_point + ts * to_point


GRID_SIZE = 1000
RADIUS = 50
FRAMES = 20
SEED = 1701

# Generate the path and eased positions/velocities
rng = np.random.default_rng(seed=SEED)
path = rng.integers(RADIUS, GRID_SIZE - RADIUS, size=(5, 2, 1))
path = np.concat([path, path[[0]]])

fracs = np.linspace(0, 1, FRAMES)
eased_positions = ease_inoutsine(fracs)
eased_velocties = 1 + (np.diagonal(jacobian(ease_inoutsine)(fracs)) ** 3)

# Iterate along the path, generating and storing frames on the grid
ims = []
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
grid = np.zeros((GRID_SIZE, GRID_SIZE))
for p1, p2 in zip(path, path[1:]):
    positions = lerp(p1, p2, eased_positions)
    for (y, x), velocity in zip(zip(*positions), eased_velocties):
        grid *= 0.8
        rr, cc = ellipse(
            r=y,
            c=x,
            r_radius=(RADIUS / velocity),
            c_radius=(RADIUS * velocity),
            shape=grid.shape,
            rotation=-np.atan2(*(p2 - p1).flatten()),
        )
        grid[rr, cc] = 1
        im = ax.imshow(grid, cmap="gray")
        ims.append([im])

# Create and save animation
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat=True)
ani.save("out/day02.gif", fps=60, savefig_kwargs=dict(transparent=True))
plt.close()
