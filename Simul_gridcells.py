

# interference model
import numpy as np
import Utility

n = 45
angle_shift = 0
grid_scale = 10

# store the 2d grid patterns in each of 45-by-45 locations
grid_pattern_array = np.zeros((n, n, n, n))
for loc_x in list(range(0, n)):
    for loc_y in list(range(0, n)):
        orientation = 0 + angle_shift
        wave1 = Utility.gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
        orientation = 60 + angle_shift
        wave2 = Utility.gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
        orientation = 120 + angle_shift
        wave3 = Utility.gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
        grid_pattern = np.exp((wave1 + wave2 + wave3))
        grid_pattern_array[loc_x, loc_y, ...] = grid_pattern


# plot
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.close('all')
fig = plt.figure(figsize=(30, 30))
n = 45
gs = gridspec.GridSpec(n, n)
gs.update(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
for loc_x in list(range(0, n)):
    for loc_y in list(range(0, n)):
        img = grid_pattern_array[loc_x, loc_y].reshape(n, n)
        ax = fig.add_subplot(gs[loc_x, loc_y])
        ax.imshow(img, cmap='jet')
        ax.axis('off')
plt.show()

