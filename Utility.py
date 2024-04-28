from os import listdir
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage as ndimage
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import moviepy.video.io.ImageSequenceClip
import imageio as iio


# simulate ideal C_location
def access_distance_map(n, goal_loc):
    distance_map = np.zeros((n, n))
    for ith_x in list(range(0, n)):
        for ith_y in list(range(0, n)):
            distance_map[ith_x, ith_y] = np.sqrt(np.abs(goal_loc[1] - ith_y) ** 2 + np.abs(goal_loc[0] - ith_x) ** 2)
    distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))
    distance_map = 1 - distance_map
    return distance_map


# transform 2d coordinate to 1d
def index_2d_to_1d(x, y, n):
    return x * n + y

# transform 1d coordinate to 2d
def index_1d_to_2d(index_1d, n):
    return np.array([int(index_1d / n), int(index_1d % n)])

# compute angle between 2 directions
def minimum_diff_angle(angle, ref_direction):
  res = (angle-ref_direction)%180
  if res < 90:
    return res
  else:
    return (180-res)

# compute mean direction
def compute_mean_angle(angle_vector):
    x = y = 0
    for ith_ori in list(range(0, len(angle_vector))):
        cur_ori = angle_vector[ith_ori]
        x += np.cos(cur_ori*np.pi/180)
        y += np.sin(cur_ori*np.pi/180)
    return np.round(np.remainder(np.arctan2(y, x) * 180/np.pi, 360))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# return coordinate list between two coordinate ([x1, y1] and [x2, y2])
def intermediates(p1, p2, nb_points):
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)
    out_x = np.array([p1[0] + i * x_spacing for i in range(1, nb_points + 1)])
    out_y = np.array([p1[1] + i * y_spacing for i in range(1, nb_points + 1)])
    out_x = out_x.astype(int)
    out_y = out_y.astype(int)
    # path_index_1d = index_2d_to_1d(out_x, out_y, n)
    return out_x, out_y

# return list of start locations
def obtain_start_locations(n, type, radius):
    if type == 'circle':
        phase_list = np.linspace(0, np.pi * 2, 360, endpoint=True)
        phase_x_list = np.cos(phase_list) * radius + n / 2
        phase_y_list = np.sin(phase_list) * radius + n / 2
        phase_x_index = [int(x) for x in phase_x_list]
        phase_y_idnex = [int(x) for x in phase_y_list]
        loc_list = np.zeros((360, 2)).astype(int)
        loc_list[:, 0] = phase_x_index
        loc_list[:, 1] = phase_y_idnex
    if type == 'square':
        square_space = np.zeros((n, n))
        for ith_x in list(range(0, n)):
            for ith_y in list(range(0, n)):
                if ith_x == 0 or ith_x == n - 1 or ith_y == 0 or ith_y == n - 1:
                    square_space[ith_x, ith_y] = 1
        index_pool = np.where(square_space == 1)
        loc_list = np.zeros((np.shape(index_pool)[1], 2)).astype(int)
        loc_list[:, 0] = index_pool[0]
        loc_list[:, 1] = index_pool[1]
    return loc_list


def plot_path():
    mypath_traj = '/Users/bo/Desktop/temp_traj/'
    mypath_grid = '/Users/bo/Desktop/temp_grid/'
    file_list_traj = np.array([mypath_traj + f for f in listdir(mypath_traj) if "counter" in f])
    file_list_grid = np.array([mypath_grid + f for f in listdir(mypath_grid) if "counter" in f and "scale33" in f])
    corr_index_list_traj = [int(re.search('counter(.*)_traj', f).group(1)) for f in file_list_traj]
    corr_index_list_grid = [int(re.search('counter(.*)_traj', f).group(1)) for f in file_list_grid]
    file_list_traj = file_list_traj[np.argsort(corr_index_list_traj)]
    file_list_grid = file_list_grid[np.argsort(corr_index_list_grid)]
    for ith in list(range(0, len(file_list_traj))):
        plt.close('all')
        fig = plt.figure(figsize=(13, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], height_ratios=[1])
        gs.update(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
        img_traj = iio.v3.imread(file_list_traj[ith])
        img_grid = iio.v3.imread(file_list_grid[ith])
        ax = fig.add_subplot(gs[0])
        ax.imshow(img_traj, cmap='jet')
        ax.axis('off')
        ax = fig.add_subplot(gs[1])
        ax.imshow(img_grid, cmap='jet')
        ax.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig('/Users/bo/Desktop/temp_combine/counter' + str(ith) + '.jpg')

def make_video():
    mypath = '/Users/bo/Desktop/temp/'
    file_list = np.array([mypath + f for f in listdir(mypath) if "angle" in f])
    corr_index_list = [int(re.search('angle(.*).png', f).group(1)) for f in file_list]
    file_list = file_list[np.argsort(corr_index_list)]
    fps = 10
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(file_list.tolist(), fps=fps)
    clip.write_videofile('/Users/bo/Desktop/temp.mp4')


# generate planar wave (grid orientation, spatial scale, loc_x, loc_y, n)
def gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n):
    resolution = 1
    amplitude = 1/3
    x = np.arange(0, n, resolution) - loc_x  # (loc_x / n) * cur_scale
    y = np.arange(0, n, resolution) - loc_y  # (loc_y / n) * cur_scale
    [xx, yy] = np.meshgrid(x, y)
    kx = np.cos(orientation * np.pi / 180)
    ky = np.sin(orientation * np.pi / 180)
    w = 2 * np.pi / (grid_scale)
    gradient_map = w * (kx * xx + ky * yy)
    plane_wave = amplitude * (np.cos(gradient_map))
    return plane_wave

def access_grid_pattens(angle_shift,grid_scale, n):
    # store the 2d grid patterns in each of 45-by-45 locations
    grid_pattern_array = np.zeros((n, n, n, n))
    for loc_x in list(range(0, n)):
        for loc_y in list(range(0, n)):
            orientation = 0 + angle_shift
            wave1 = gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
            orientation = 60 + angle_shift
            wave2 = gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
            orientation = 120 + angle_shift
            wave3 = gen_2d_wave(orientation, loc_x, loc_y, grid_scale, n)
            grid_pattern = np.exp((wave1 + wave2 + wave3))
            grid_pattern_array[loc_x, loc_y, ...] = grid_pattern
    return grid_pattern_array




