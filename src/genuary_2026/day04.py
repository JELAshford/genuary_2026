"""Lowres! Low resolution...low resolution..."""

import matplotlib.pylab as plt
from PIL import Image
import numpy as np

SCALE = 12
NUM_RECTS = 20
SEED = 1701

# Load image
image = Image.open("rsc/thames_view.jpg")
new_shape = np.ceil(np.array([image.width, image.height]) / SCALE) * SCALE
image = image.resize(new_shape.astype(int))
image_array = np.array(image)
height, width, _ = image_array.shape

# Make low-res
lowres = image_array[::SCALE, ::SCALE, :].repeat(SCALE, 0).repeat(SCALE, 1)

# Create random squares of high res
base, top = lowres, image_array

# Transition between the low and high res options
rng = np.random.default_rng(seed=SEED)
point = np.array([height // 2, width // 2]).astype(int)
yy, xx = np.meshgrid(np.arange(width), np.arange(height))
dist = yy.astype(float)
dist /= dist.max()
dist += rng.random(dist.shape) * 0.5
dist = dist[::SCALE, ::SCALE].repeat(SCALE, 0).repeat(SCALE, 1)
mix = dist > 0.7
# plt.imshow(mix)
# plt.show()
base[mix] = top[mix]

# Show!
fig, ax = plt.subplots(1, 1)
ax.imshow(base)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day04.png", dpi=200, bbox_inches="tight", transparent=True)
plt.show()
