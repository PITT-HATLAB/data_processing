# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:27:01 2021

@author: Hatlab_3
"""

# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target, log_normalize_up_to_row
import matplotlib.colors as color
from scipy.ndimage import gaussian_filter

#Get Taco (irregular imshow)
def make_tacos(bias_current, gen_frequency, gen_power, calculated_gain, replace_nan = False, vmin = 15, vmax = 25, fancy = False): 
    fig, ax = plt.subplots(1,1)
    if replace_nan: 
        calculated_gain[np.isnan(calculated_gain)] = 0
    img = ax.scatter(gen_frequency/1e6, gen_power, c = calculated_gain, cmap = 'seismic', vmin = vmin, vmax = vmax, zorder = 1)
    cb = fig.colorbar(img, ax = ax)
    unique_freqs = np.unique(gen_frequency)
    best_powers = [select_closest_to_target(gen_power[gen_frequency == f], calculated_gain[gen_frequency == f], 20) for f in unique_freqs]
    ax.plot(unique_freqs/1e6, best_powers, 'k-', lw = 2)
    return fig, ax, cb

def make_sat_img_plot(sat_bias_current, sat_gen_freq, sat_vna_powers, sat_gain, levels = [-2,-1.5,-1, -0.25, 0.25,1, 1.5,2], norm_power = -40, x_val = None, filter_window = 0, vmin = -1, vmax = 1): 
    y_norm_val = norm_power #Signal power at which to normalize the rest of the plot to
    # print(f"Normalized to VNA_Power = {y_norm_val}dB")
    fig, ax = plt.subplots(1,1)
    colors = [color.hex2color('#4444FF'), color.hex2color('#FFFFFF'), color.hex2color('#888888'), color.hex2color('#888888'),color.hex2color('#FFFFFF'), color.hex2color('#FF4444')]
    _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
    
    smoothed_normed_data = log_normalize_to_row(sat_gen_freq, sat_vna_powers[0], gaussian_filter(sat_gain.T, (filter_window,0)), y_norm_val= y_norm_val)
    img = ax.pcolor(sat_gen_freq/1e6, sat_vna_powers[0], 
                      smoothed_normed_data, 
                      # gaussian_filter(sat_gain.T, 5), 
                      cmap = _cmap, 
                      vmin = vmin, vmax = vmax)
    #getting saturation points
    sat_powers = []
    for col in smoothed_normed_data.T: 
        buffer = np.size(col[sat_vna_powers[0]<= y_norm_val])
        #append locations of +1dB and -1dB points
        try: 
            pos_loc = buffer+np.min(np.where(np.isclose(col[sat_vna_powers[0]>y_norm_val], 1, atol = 1e-2))[0])
        except ValueError: 
            pos_loc = np.size(col)-1
        try: 
            neg_loc = buffer+np.min(np.where(np.isclose(col[sat_vna_powers[0]>y_norm_val], -1, atol = 1e-2))[0])
        except ValueError: 
            neg_loc = np.size(col)-1
            
        # print(f"Pos: {pos_loc} \nNeg: {neg_loc}")
        loc_arr = np.array([pos_loc, neg_loc])
        loc_arr = np.floor(loc_arr[np.logical_not(np.isnan(loc_arr))]).astype(int)
        # print(loc_arr)
        loc = np.min(loc_arr)
        sat_powers.append(sat_vna_powers[0][loc])
    plt.plot((np.array(sat_gen_freq+(sat_gen_freq[1]-sat_gen_freq[0])/2)/1e6)[0:-1], sat_powers[0:-1], 'k o')
    
    #plot the best one as a star
    max_loc = np.where(sat_powers[0:-1] == np.max(sat_powers[0:-1]))[0][0]
    print(max_loc)
    plt.plot((np.array(sat_gen_freq+(sat_gen_freq[1]-sat_gen_freq[0])/2)/1e6)[max_loc], sat_powers[max_loc], 'r*', markersize = 5)
    ax.hlines(y = y_norm_val, xmin = np.min(sat_gen_freq/1e6), xmax = np.max(sat_gen_freq/1e6), color = 'b', lw = 2)
    return fig, ax, img