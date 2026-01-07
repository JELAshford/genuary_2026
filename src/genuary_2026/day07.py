"""Boolean Algebra! Release the bitfields!!!"""

import matplotlib.pylab as plt
import numpy as np

SIZE = 2000
yy, xx = np.meshgrid(range(SIZE), range(SIZE))

# Show!
fig, ax = plt.subplots(1, 1)
ax.imshow((yy ^ xx) % xx % yy - (xx % (yy / 30)), cmap="Blues")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("out/day07.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
