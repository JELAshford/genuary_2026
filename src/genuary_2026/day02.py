"""Twelve principles of animation! Lerp and squish
Easing resource: https://easings.net
"""

import matplotlib.animation as animation
from skimage.draw import ellipse
import matplotlib.pylab as plt
from autograd import jacobian
import autograd.numpy as np


def scale_and_offset(frac_array: np.ndarray, min_val: int, max_val: int):
    return (frac_array * (max_val - min_val)) + min_val


def ease_inoutsine(t: np.ndarray):
    return -(np.cos(np.pi * t) - 1) / 2


GRID_SIZE = 1000
CIRCLE_RADIUS = 100
FRAMES = 20

fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])

# Basics: let's make an animation of a circle going back and forht
# then we can add the squooshing and easing
grid = np.zeros((GRID_SIZE // 2, GRID_SIZE))
ims = []
fracs = np.linspace(0, 1, FRAMES)
eased_positions = ease_inoutsine(ease_inoutsine(fracs))
eased_velocties = 0.8 + (np.diagonal(jacobian(ease_inoutsine)(fracs)) ** 3 / 2)
positions = scale_and_offset(
    eased_positions,
    min_val=CIRCLE_RADIUS * 0.8,
    max_val=GRID_SIZE - (CIRCLE_RADIUS * 0.8),
)
eased_velocties = np.concat([eased_velocties, eased_velocties[::-1][1:]])
positions = np.concat([positions, positions[::-1][1:]])
for position, velocity in zip(positions, eased_velocties):
    grid *= 0.8
    # grid = np.zeros((GRID_SIZE // 2, GRID_SIZE))
    row = GRID_SIZE // 4
    r_radius = (CIRCLE_RADIUS * 1.5) / (velocity * 2)
    c_radius = (CIRCLE_RADIUS * 1.5) * (velocity * 0.5)
    rr, cc = ellipse(
        r=row,
        c=position,
        r_radius=r_radius,
        c_radius=c_radius,
        shape=grid.shape,
    )
    grid[rr, cc] = 1
    # rr, cc = ellipse(
    #     r=row,
    #     c=position,
    #     r_radius=r_radius // 2,
    #     c_radius=c_radius // 2,
    #     shape=grid.shape,
    # )
    # grid[rr, cc] = 0
    im = ax.imshow(grid, cmap="gray")
    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=10,
    blit=True,
    repeat=True,
)
ani.save("out/day02.gif", fps=60)

plt.show()
