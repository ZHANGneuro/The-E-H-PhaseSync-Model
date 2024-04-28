
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Utility
import scipy
import os
from shutil import rmtree
import random
import scipy.ndimage as ndimage

# space size, we used 45 in paper, and here increase the space size for visualization purpose
n = 80
grid_orientation = 45

# goal location
goal_loc = [int(n/2), int(n/2)]

# generate orientation coordiate in dimension: 45 (orientation coordinate) by 2 (x & y) by 24 (location samples)
orientation_samples = 60
ori_x = np.cos(np.linspace(0, np.pi, orientation_samples, endpoint=False)) * (n/2)*4/5 + (n/2)
ori_x = np.round(ori_x).astype(int)
ori_y = np.sin(np.linspace(0, np.pi, orientation_samples, endpoint=False)) * (n/2)*4/5 + (n/2)
ori_y = np.round(ori_y).astype(int)
coordinate_ori = np.zeros((n, 2, orientation_samples)).astype(int)
for ind_ori in list(range(0, len(ori_x))):
    # cur_index = Utility.index_2d_to_1d(ori_x[ind_ori], ori_y[ind_ori], n=n)
    coordinate_ori[:, 0, ind_ori] = Utility.intermediates(p1=[ori_x[ind_ori], ori_y[ind_ori]], p2=[n-ori_x[ind_ori], n-ori_y[ind_ori]], nb_points=n)[0]
    coordinate_ori[:, 1, ind_ori] = Utility.intermediates(p1=[ori_x[ind_ori], ori_y[ind_ori]], p2=[n-ori_x[ind_ori], n-ori_y[ind_ori]], nb_points=n)[1]

# plt.close('all')
# fig = plt.figure(figsize=(7, 7))
# ax4 = fig.add_subplot()
# loc_map = np.zeros((n,n))
# loc_map[ori_x, ori_y] = 1
# ax4.imshow(loc_map, cmap = matplotlib.colors.ListedColormap(['black', 'red']))
# plt.show()
# plt.close('all')
# fig = plt.figure(figsize=(7, 7))
# ax4 = fig.add_subplot()
# loc_map = np.zeros((n,n))
# loc_map[n-ori_x, n-ori_y] = 1
# ax4.imshow(loc_map, cmap = matplotlib.colors.ListedColormap(['black', 'red']))
# plt.show()

# access grid pattern array in dimension [45, 45, 45, 45], which is [loc_x, loc_y, grid_code_x, grid_code_y]
grid_array = Utility.access_grid_pattens(angle_shift=grid_orientation, grid_scale=30, n=n)

# folder to store files for checking orientation representations
folder_check = './test_ori_rep'
if os.path.exists(folder_check):
    rmtree(folder_check)
os.mkdir(folder_check)

# create delta vector
delta_list = []
for ind_ori in list(range(0, orientation_samples)):

    mean_grid_patten = np.mean(grid_array[coordinate_ori[:,0,ind_ori], coordinate_ori[:, 1, ind_ori], :, :], axis=0)

    max_list = []
    for ind_find_max in list(range(0, orientation_samples)):
        max_list.append(np.mean(mean_grid_patten[coordinate_ori[:,0,ind_find_max], coordinate_ori[:, 1, ind_find_max]]))
    delta_list.append(np.sum(mean_grid_patten[coordinate_ori[:,0,np.argmax(max_list)], coordinate_ori[:, 1, np.argmax(max_list)]]))

    # plot path code V
    plt.close('all')
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0)
        ax.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
    plt.imshow(mean_grid_patten, cmap='jet',alpha=1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(folder_check + '/Ori_' + str(ind_ori) + '_V.png')

    # plot orientation coordinate
    ori_coor = np.zeros((n,n))
    ori_coor[coordinate_ori[:,0,np.argmax(max_list)], coordinate_ori[:, 1, np.argmax(max_list)]] = 1
    plt.close('all')
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0)
        ax.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
    plt.imshow(ori_coor, cmap = matplotlib.colors.ListedColormap(['black', 'red']))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(folder_check + '/Ori_' + str(ind_ori) + '_Coor.png')


# plot the delta vector
plt.close('all')
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0)
    ax.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
plt.plot(delta_list)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.savefig('/Users/bo/PycharmProject/Liu_HPC_to_ERC_model_github/test/' + str(ith) + '.png')
plt.show()


# access the periodic scaffold and plot
r_dir_fourier = np.abs(np.fft.fft(delta_list))
r_dir_fourier = r_dir_fourier[1:10]
plt.close('all')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
plt.xticks(list(range(0, 9)), list(range(1, 10)))
plt.ylabel('FFT magnitude')
plt.xlabel('#-fold')
for axis in ['top', 'right']:
    ax.spines[axis].set_linewidth(0)
    ax.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
plt.plot(r_dir_fourier, linewidth=5, color='black')
plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.2)
# plt.savefig('/Users/bo/PycharmProject/Liu_HPC_to_ERC_model_github/test/' + str(ith) + '.png')
plt.show()



# generate C_dir
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
x, y = np.meshgrid(x, y)
theta = np.arctan2(y, x)
c_dir = np.cos(3* theta + grid_orientation-90) #
c_dir = (c_dir-np.min(c_dir))/(np.max(c_dir)-np.min(c_dir))

# plot C_dir
plt.close('all')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.imshow(c_dir, cmap='jet')
plt.show()

# generate C and plot
c_dis = Utility.access_distance_map(n, goal_loc)
C = c_dir + c_dis
plt.close('all')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.imshow(C, cmap='jet')
plt.show()









# E-H phaseSync model simulation

# define number of path and initial movement step size
num_start_locs = 60
step_size = 1

# define start locations
coor_circle_x = np.cos(np.linspace(0, 2*np.pi, num_start_locs, endpoint=False)) * (n/2)*4/5 + (n/2)
coor_circle_x = np.round(coor_circle_x).astype(int)
coor_circle_y = np.sin(np.linspace(0, 2*np.pi, num_start_locs, endpoint=False)) * (n/2)*4/5 + (n/2)
coor_circle_y = np.round(coor_circle_y).astype(int)

# define the number of actions considered to move for each location
num_actions = 24
matrix_locs = np.concatenate((coor_circle_x.reshape(len(coor_circle_x), 1), coor_circle_y.reshape(len(coor_circle_y), 1)), axis=1)
start_locs = np.unique(matrix_locs, axis=0)
start_indexes = np.linspace(0, len(start_locs), len(start_locs), endpoint=False).astype(int)

# folder to store files for checking paths
folder_check = './test_path/'
if os.path.exists(folder_check):
    rmtree(folder_check)
os.mkdir(folder_check)

for ind_dir in list(range(0, len(start_locs[:, 0]))):
    # select a random start location
    dir_index = random.choice(start_indexes)
    start_indexes = np.delete(start_indexes, np.where(start_indexes == dir_index))
    cur_loc = np.array([start_locs[dir_index][0], start_locs[dir_index][1]]).astype(int)

    # compute the direction relative to goal
    alloc_dir = np.round(np.remainder(np.arctan2(goal_loc[1] - cur_loc[1], goal_loc[0] - cur_loc[0]) * 180 / np.pi, 360)).astype(int)

    # create a list to store movement coordiante
    record_trajectory = []
    record_trajectory.append(cur_loc)

    # start movement using winner-take-all dynamics
    for ith_movement in range(500):
        while True:
            old_consink = C[cur_loc[0], cur_loc[1]]
            hd_dirs = np.linspace(0, 360, num_actions, endpoint=False).astype(int) * np.pi / 180
            circle_x = np.cos(hd_dirs) * step_size
            circle_y = np.sin(hd_dirs) * step_size
            matrix_locs = np.concatenate((circle_x.astype(int).reshape(num_actions,1), circle_y.astype(int).reshape(num_actions,1)), axis=1)

            action_list = np.zeros((num_actions))
            next_loc_list = (cur_loc + matrix_locs).astype(int)
            for ith_action in list(range(0, matrix_locs.shape[0])):
                cur_coor = next_loc_list[ith_action]
                if np.min(cur_coor) >= 0 and np.max(cur_coor) < n:
                    action_list[ith_action] = C[cur_coor[0], cur_coor[1]] #
                else:
                    action_list[ith_action] = -1

            Max_index = np.argmax(action_list)
            if old_consink<action_list[Max_index]:
                cur_loc = next_loc_list[Max_index]
                break
            else:
                step_size += 1

        # store the coordinate
        record_trajectory.append(cur_loc)

        # check if the path is finished
        if np.sqrt((cur_loc[0] - goal_loc[0]) ** 2 + (cur_loc[1] - goal_loc[1]) ** 2) <= 5:
            print('reach the goal at' + str(ind_dir))
            break

    np.save(folder_check + 'file_trajectory_ith' + str(ind_dir) + '_dir' + str(alloc_dir) + '.npy', record_trajectory)


# test 2d trajectory
import numpy as np
import matplotlib.pyplot as plt
import re
import moviepy.video.io.ImageSequenceClip
from os import listdir
import imageio as iio
from matplotlib import gridspec
import matplotlib
traj_list = np.array([folder_check + f for f in listdir(folder_check) if "file_trajectory" in f])
corr_index_list = [int(re.search('ith(.*)_dir', f).group(1)) for f in traj_list]
traj_list = traj_list[np.argsort(corr_index_list)].tolist()

plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 30}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
output_traj = []
for ith in list(range(0, len(traj_list))):
    cur_file = np.load(traj_list[ith])
    output_traj.append(cur_file)
    ax.plot(cur_file[:,0], cur_file[:,1], color='black', linewidth = 5)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(30)
    ax.spines[axis].set_color((0 / 255, 0 / 255, 0 / 255))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()








# plot all
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, wspace=0.5, hspace=0.8, left=0.05, right=0.95, top=0.9, bottom=0.05)

ax0 = fig.add_subplot(gs[0])
ax0.set_title('delta vector')
ax0.set_ylabel('delta')
ax0.set_xlabel('Spatial orientation')
for axis in ['top', 'right']:
    ax0.spines[axis].set_linewidth(0)
    ax0.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
ax0.plot(delta_list, color='black', linewidth = 3)

ax1 = fig.add_subplot(gs[1])
ax1.set_title('FFT output')
ax1.set_xticks(list(range(0, 9)), list(range(1, 10)))
ax1.set_ylabel('FFT magnitude')
ax1.set_xlabel('#-fold')
for axis in ['top', 'right']:
    ax1.spines[axis].set_linewidth(0)
    ax1.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
ax1.plot(r_dir_fourier, linewidth=3, color='black')

ax2 = fig.add_subplot(gs[2])
ax2.set_title('C_dir')
ax2.set_ylabel('Loogit')
ax2.set_xlabel('Vacso')
for axis in ['top', 'right']:
    ax2.spines[axis].set_linewidth(0)
    ax2.spines[axis].set_color((191 / 255, 189 / 255, 132 / 255))
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
ax2.imshow(c_dir, cmap='jet')

ax3 = fig.add_subplot(gs[3])
ax3.set_title('C')
ax3.set_ylabel('Loogit')
ax3.set_xlabel('Vacso')
for axis in ['top', 'right', 'bottom', 'left']:
    ax3.spines[axis].set_linewidth(0)
ax3.get_xaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
ax3.imshow(C, cmap='jet')

ax4 = fig.add_subplot(gs[4])
ax4.set_title('Start-locations')
for axis in ['top', 'right', 'bottom', 'left']:
    ax4.spines[axis].set_linewidth(0)
ax4.get_xaxis().set_ticks([])
ax4.get_yaxis().set_ticks([])
ax4.set_ylabel('Loogit')
ax4.set_xlabel('Vacso')
loc_map = np.zeros((n,n))
loc_map[start_locs[:,0], start_locs[:,1]] = 1
ax4.imshow(loc_map, cmap = matplotlib.colors.ListedColormap(['black', 'red']))

ax5 = fig.add_subplot(gs[5])
ax5.set_title('model simulation')
ax5.set_ylabel('Loogit')
ax5.set_xlabel('Vacso')
for axis in ['top', 'right', 'bottom', 'left']:
    ax5.spines[axis].set_linewidth(0)
ax5.get_xaxis().set_ticks([])
ax5.get_yaxis().set_ticks([])
for ith in list(range(0, len(traj_list))):
    cur_file = np.load(traj_list[ith])
    ax5.plot(cur_file[:,0], cur_file[:,1], color='black', linewidth = 2)
    ax5.set_xlim(0, n)
    ax5.set_ylim(0, n)
plt.savefig('./model_output.png')







