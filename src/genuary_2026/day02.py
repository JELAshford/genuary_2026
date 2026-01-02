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

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])

# Generate x-axis positions and velocities for the back-and-forth motion
path = np.array([[GRID_SIZE // 4, RADIUS], [GRID_SIZE // 4, GRID_SIZE - RADIUS]]).T
fracs = np.linspace(0, 1, FRAMES)
eased_positions = ease_inoutsine(fracs)
eased_velocties = 1 + (np.diagonal(jacobian(ease_inoutsine)(fracs)) ** 3)
positions = lerp(path[:, [0]], path[:, [1]], eased_positions)

looped_positions = np.concat([positions, positions[:, ::-1][:, 1:]], axis=-1)
looped_velocties = np.concat([eased_velocties, eased_velocties[::-1][1:]])

# Draw the stretched circle and save Artist objects
ims = []
grid = np.zeros((GRID_SIZE // 2, GRID_SIZE))
for (y, x), velocity in zip(zip(*looped_positions), looped_velocties):
    grid *= 0.8
    rr, cc = ellipse(
        r=y,
        c=x,
        r_radius=(RADIUS / velocity),
        c_radius=(RADIUS * velocity),
        shape=grid.shape,
    )
    grid[rr, cc] = 1
    im = ax.imshow(grid, cmap="gray")
    ims.append([im])

# Create and save animation
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat=True)
ani.save("out/day02.gif", fps=60, savefig_kwargs=dict(transparent=True))
plt.close()
