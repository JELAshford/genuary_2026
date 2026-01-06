"""Write “Genuary”. Avoid using a font."""

import matplotlib.pylab as plt
from einops import repeat
from PIL import Image
import numpy as np

SEED = 1702
NUM_POINTS = 1500

rng = np.random.default_rng(seed=SEED)

# Load base image of text
text = Image.open("rsc/genuary_mask.png").convert("L")
text_array = (255 - np.array(text)) / 255
text_array = text_array[::3, ::3]
height, width = text_array.shape

# Add all the points
possible_positions = np.argwhere(text_array == 1)
ball_points = rng.choice(possible_positions, size=NUM_POINTS)

# Place the points then grow them!
out = np.zeros(text_array.shape)
for step in range(10):
    out *= 0.8
    ball_points = ball_points + rng.integers(-1, 2, size=ball_points.shape)
    ball_points[:, 0] %= height
    ball_points[:, 1] %= width
    out[*ball_points.T] += 10

# Channel shenanigans
out = repeat(out, "h w -> h w 3")
out[..., 0] = np.roll(out[..., 0], 2, 0)
out[..., 1] = np.roll(out[..., 1], -2, 1)
out[..., 2] = np.roll(out[..., 1], -2, 0)

# Show!
fig, ax = plt.subplots(1, 1)
ax.imshow(out)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day05.png", dpi=200, bbox_inches="tight", transparent=True)
plt.show()
