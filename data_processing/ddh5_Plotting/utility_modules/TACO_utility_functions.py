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

def make_gain_profiles(filepath, replace_nan = False, vmin = 15, vmax = 25, angles = [45, 45]):
    
    gainDict = all_datadicts_from_hdf5(filepath)['data']
    calc_gain = gainDict.extract('calculated_gain').data_vals('calculated_gain')
    gen_frequency_calc = gainDict.extract('calculated_gain').data_vals('gen_frequency')
    gen_power_calc = gainDict.extract('calculated_gain').data_vals('gen_power')
    unique_freqs = np.unique(gen_frequency_calc)
    best_powers = [select_closest_to_target(gen_power_calc[gen_frequency_calc == f], calc_gain[gen_frequency_calc == f], 20) for f in unique_freqs]
    
    gain_traces = gainDict.extract('gain_trace').data_vals('gain_trace')
    gen_power = gainDict.extract('gain_trace').data_vals('gen_power')
    vna_freqs = gainDict.extract('gain_trace').data_vals('vna_frequency')
    gen_frequency_raw = gainDict.extract('gain_trace').data_vals('gen_frequency')
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.azim = angles[0]
    ax.elev = angles[1]
    
    for best_power in best_powers:     
        gp_filt = np.isclose(gen_power, best_power, atol = 0.05)
        f_val= np.round(np.average(gen_frequency_raw[gp_filt])/1e6, 0)
        ax.plot(f_val*np.ones(np.size(vna_freqs[gp_filt][0])), vna_freqs[gp_filt][0]/1e6, gain_traces[gp_filt][0])
        
    return fig, ax

def make_gain_surface(filepath, replace_nan = False, vmin = 15, vmax = 25, angles = [45, 45]):
    
    gainDict = all_datadicts_from_hdf5(filepath)['data']
    calc_gain = gainDict.extract('calculated_gain').data_vals('calculated_gain')
    gen_frequency_calc = gainDict.extract('calculated_gain').data_vals('gen_frequency')
    gen_power_calc = gainDict.extract('calculated_gain').data_vals('gen_power')
    unique_freqs = np.unique(gen_frequency_calc)
    best_powers = [select_closest_to_target(gen_power_calc[gen_frequency_calc == f], calc_gain[gen_frequency_calc == f], 20) for f in unique_freqs]
    
    gain_traces = gainDict.extract('gain_trace').data_vals('gain_trace')
    gen_power = gainDict.extract('gain_trace').data_vals('gen_power')
    vna_freqs = gainDict.extract('gain_trace').data_vals('vna_frequency')
    gen_frequency_raw = gainDict.extract('gain_trace').data_vals('gen_frequency')
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.azim = angles[0]
    ax.elev = angles[1]
    
    gen_f_array = []
    sig_f_array = []
    gain_array = []
    for best_power in best_powers:     
        gp_filt = np.isclose(gen_power, best_power, atol = 0.05)
        f_val= np.round(np.average(gen_frequency_raw[gp_filt])/1e6, 0)
        
        gen_f_array.append(f_val*np.ones(np.size(vna_freqs[gp_filt][0])))
        sig_f_array.append(vna_freqs[gp_filt][0]/1e6)
        gain_array.append(gain_traces[gp_filt][0])
    
    gen_f_array = np.array(gen_f_array)
    sig_f_array = np.array(sig_f_array)
    gain_array = np.array(gain_array)
    
    ax.plot_surface(gen_f_array, sig_f_array, gain_array, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    
    return fig, ax

def make_sat_img_plot(sat_bias_current, sat_gen_freq, sat_vna_powers, sat_gain, levels = [-2,-1.5,-1, -0.25, 0.25,1, 1.5,2], norm_power = -40, x_val = None, filter_window = 0, vmin = -1, vmax = 1): 
    y_norm_val = norm_power #Signal power at which to normalize the rest of the plot to
    # print(f"Normalized to VNA_Power = {y_norm_val}dB")
    fig, ax = plt.subplots()
    colors = [color.hex2color('#4444FF'), color.hex2color('#FFFFFF'), color.hex2color('#888888'), color.hex2color('#888888'),color.hex2color('#FFFFFF'), color.hex2color('#FF4444')]
    _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
    
    smoothed_normed_data = log_normalize_to_row(sat_gen_freq, sat_vna_powers[0], gaussian_filter(sat_gain.T, (filter_window,0)), y_norm_val= y_norm_val)
    img = ax.pcolormesh(sat_gen_freq/1e6, sat_vna_powers[0], 
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
        
    ax.plot((np.array(sat_gen_freq+(sat_gen_freq[1]-sat_gen_freq[0])/2)/1e6)[0:-1], sat_powers[0:-1], 'k o')
    #plot the best one as a star
    max_loc = np.where(sat_powers[0:-1] == np.max(sat_powers[0:-1]))[0][0]
    # print(max_loc)
    plt.plot((np.array(sat_gen_freq+(sat_gen_freq[1]-sat_gen_freq[0])/2)/1e6)[max_loc], sat_powers[max_loc], 'r*', markersize = 5)
    ax.hlines(y = y_norm_val, xmin = np.min(sat_gen_freq/1e6), xmax = np.max(sat_gen_freq/1e6), color = 'b', lw = 2)
    return fig, ax, img

def superTACO_Lines(filepaths, angles = [45,45], quanta_size = None, quant_offset = None):
    #step 1: assemble best powers into bias_currents vs. (gen_freq vs. best_powers) array
    #ie for each n bias current there is a gen_freq array
    #and for each m(n) gen_freq there is one gen_power that is best (could be NaN if garbage)
    #feed this into n mplot3d commands each with their own oclor and legend label
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.azim = angles[0]
    ax.elev = angles[1]
    
    bias_currents = []
    best_gen_frequencies = []
    best_gen_powers = []
    
    for gain_filepath in filepaths: 
        #extract the best gen powers
        gain_dicts = all_datadicts_from_hdf5(gain_filepath)
        gainDict = gain_dicts['data']
        gain_data = gainDict.extract('calculated_gain')
        [bias_current, gen_frequency, gen_power, calc_gain] = [gain_data.data_vals('bias_current'),
                                                               gain_data.data_vals('gen_frequency'), 
                                                               gain_data.data_vals('gen_power'), 
                                                               gain_data.data_vals('calculated_gain')
                                                               ]
        
        for current in np.unique(bias_current): #could be multiple bias currents in one single TACO datafile
            bias_currents.append(current)
            print(f"{gain_filepath}\nCURRENT: {current*1000}mA")
            filt = bias_current == current
            cfreqs = gen_frequency[filt]
            cpowers = gen_power[filt]
            unique_freqs = np.unique(cfreqs)
            cgain = calc_gain[filt]
            
            best_powers = [select_closest_to_target(cpowers[cfreqs == f], cgain[cfreqs == f], 20) for f in unique_freqs]
            #convert freqs to detuning from best power
            best_power = np.min(best_powers)
            best_gen_powers.append(best_power)

            best_freq = np.average(unique_freqs[np.where(best_powers == best_power)])
            best_gen_frequencies.append(best_freq)
            
            adjusted_freqs = unique_freqs - best_freq
            if quanta_size != None:
                quant_frac = np.round((current-quant_offset)/quanta_size, 3)
                ax.plot(current*np.ones(np.size(unique_freqs))*1000, adjusted_freqs/1e6, best_powers, label = f'{quant_frac} quanta')
                
            else: 
                ax.plot(current*np.ones(np.size(unique_freqs))*1000000, adjusted_freqs/1e6, best_powers)
    ax.set_xlabel("Bias Current (mA)")
    ax.set_ylabel("Generator Detuning (MHz)")
    ax.set_zlabel("Generator Power (dBm)")
    ax.set_title("20dB Gain Power vs. Flux Bias and Generator Detuning")
    
    return [np.array(bias_currents), np.array(best_gen_frequencies), np.array(best_gen_powers)]