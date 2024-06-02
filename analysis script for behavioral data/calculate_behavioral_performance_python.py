
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
from matplotlib import gridspec
from numpy import loadtxt
import module_utility
import scipy
import random

#
angle_colume = 14 # access angles from this colume
num_samples = 36 # number of samples
output_table = np.zeros((20, num_samples)) # array to store movement distance
error_table = np.zeros((20, num_samples)) # array to store errors (distance between ending-location and goal-location)
baseline_table = np.zeros((20,num_samples)) # array to store the baselines for movement distance

for ith_sub in list(range(1, 21)):
    # load detailed behavioral data
    cur_path_detail = '/Users/bo/Documents/data_liujia_lab/analysis_liuP1_greeble/sub' + str(ith_sub) + '_mri_detail.txt'
    raw_data = loadtxt(cur_path_detail, dtype=str)
    detail_table = np.zeros((len(raw_data), 15))
    for ith_row in list(range(0, len(raw_data))):
        cur_row = raw_data[ith_row]
        for ith_col in list(range(0, 15)):
            detail_table[ith_row, ith_col] = cur_row[ith_col]
    detail_table = detail_table.astype(int)

    # compute movement distance
    num_movement_list =[]
    for ith_sess in list(range(1, 9)):
        sess_table = detail_table[np.where(detail_table[:,1]==ith_sess)[0], ...]
        trial_list = np.unique(sess_table[:,2])
        for cur_trial in trial_list:
            num_movement_list.append(len(np.where(sess_table[:,2]==cur_trial)[0]))
    num_movement_list = np.array(num_movement_list)

    # load basic behavioral data
    cur_path_task = '/Users/bo/Documents/data_liujia_lab/analysis_liuP1_greeble/sub' + str(ith_sub) + '_mri_record_subjDir.txt'
    raw_data_task = loadtxt(cur_path_task, dtype=str)
    raw_data_task = np.delete(raw_data_task, np.where(raw_data_task[:,2]=='NA')[0], axis=0)
    angle_list = np.sort(np.unique(raw_data_task[:,angle_colume]).astype(int))
    error_list = raw_data_task[:, 7].astype(float)/0.04

    # compute baselines
    start_loc_list = []
    baseline_list = []
    for ith in list(range(0, np.shape(raw_data_task)[0])):
        cur_start_loc = [ int(float(raw_data_task[ith, 3])/0.04),  int(float(raw_data_task[ith, 4])/0.04)]
        start_loc_list.append(cur_start_loc)
        short_path = module_utility.access_shortest_path(cur_start_loc[0], cur_start_loc[1], goal_loc=[25, 25])
        baseline_list.append(len(short_path))
    baseline_list = np.array(baseline_list)

    # store movement distances, baselines, and errors into array
    for ith_angle in list(range(0, len(angle_list))):
        cur_angle = angle_list[ith_angle]
        output_table[ith_sub-1, ith_angle] = np.mean(num_movement_list[np.where(raw_data_task[:,angle_colume]==str(cur_angle))[0]])
        baseline_table[ith_sub-1, ith_angle] = np.mean(baseline_list[np.where(raw_data_task[:,angle_colume]==str(cur_angle))[0]])
        error_table[ith_sub-1, ith_angle] = np.mean(error_list[np.where(raw_data_task[:,angle_colume]==str(cur_angle))[0]])



perfor_4f_remove = ((output_table - baseline_table) + error_table) # calculate performance coefficeint
perfor_mean_across_subjects = np.mean(perfor_4f_remove, axis=0) # mean
perfor_se_across_subjects = np.std(perfor_4f_remove, axis=0)/np.sqrt(20) # standard error

# plot
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 50}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(perfor_mean_across_subjects, color='black', linewidth = 5)
ax.fill_between(range(0,perfor_4f_remove.shape[1]), perfor_mean_across_subjects-perfor_se_across_subjects, perfor_mean_across_subjects+perfor_se_across_subjects, edgecolor='black', alpha=0.3, color='gray')
ax.set_ylabel('Navigation efficiency')
for axis in ['top', 'right']:
    ax.spines[axis].set_linewidth(0)
for axis in ['left',  'bottom']:
    ax.spines[axis].set_linewidth(4)
ax.set_xticks([0, len(perfor_mean_across_subjects)-1], ['0', '2pi'])
ax.set_xlabel('Movement direction')
plt.show()



# perform fft
perfor_4f_remove = ((output_table - baseline_table)) + error_table
perfor_4f_remove = np.mean(perfor_4f_remove, axis=0)
perfor_4f_remove = scipy.signal.detrend(perfor_4f_remove)

r_dir_fourier = np.abs(np.fft.fft(perfor_4f_remove).real)
r_dir_fourier = r_dir_fourier[1:10]

# shuffling
shuf_matrix = np.zeros((5000, 9))
for ith in list(range(0, 5000)):
    dis_pool_copy = perfor_4f_remove.copy()
    random.shuffle(dis_pool_copy)
    fft_power_shuffle = np.abs(np.fft.fft(dis_pool_copy).real)
    fft_power_shuffle = fft_power_shuffle[1:10]
    shuf_matrix[ith, ...] = fft_power_shuffle
uncorr_thres_list = []
for ith in list(range(0, 9)):
    cur_list = np.sort(shuf_matrix[:,ith])
    uncorr_thres_list.append(cur_list[int(len(cur_list)*0.95)])

# plot
plt.close('all')
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 50}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(r_dir_fourier, color='black', linewidth=6)
ax.plot(list(range(0, len(r_dir_fourier))), [np.max(uncorr_thres_list)]*len(r_dir_fourier), color='r', linestyle='--', linewidth=5)
ax.set_xticks([0, 2, 4, 6, 8], [1, 3, 5 , 7, 9])
ax.set_ylabel('Spectral power')
ax.set_xlabel('#-fold modulation')
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(4)
for axis in ['top', 'right']:
    ax.spines[axis].set_linewidth(0)
plt.show()






