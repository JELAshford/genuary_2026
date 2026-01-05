"""Lowres! Low resolution...low resolution..."""

import matplotlib.pylab as plt
from PIL import Image
import numpy as np

SCALE = 16
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
# base, top = image_array, lowres

# Apply random rectangle windows
rng = np.random.default_rng(seed=SEED)
rects = rng.random(size=(NUM_RECTS, 4))
rects[:, 0] *= height
rects[:, 1] *= width
rects[:, 2] *= height // 2
rects[:, 3] *= width // 3
rects = rects.astype(int)
for y, x, h, w in rects:
    my = min(y + h, height)
    mx = min(x + w, width)
    base[(y - 5) : (my + 5), (x - 5) : (mx + 5)] = 0
    base[y:my, x:mx] = top[y:my, x:mx]

# Show!
fig, ax = plt.subplots(1, 1)
ax.imshow(base)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day04.png", dpi=200, bbox_inches="tight", transparent=True)
plt.show()
