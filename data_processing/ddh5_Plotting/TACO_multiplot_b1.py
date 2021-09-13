# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:55:36 2021

@author: Hatlab_3

Find all TACOS in a directory, pull out the best powers, and plot them in 3d space to see if it's actually a TACO or not'
"""
import os 
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target, log_normalize_up_to_row, find_all_ddh5
from mpl_toolkits.mplot3d import Axes3D
from data_processing.ddh5_Plotting.utility_modules.TACO_utility_functions import make_tacos, make_sat_img_plot
from scipy.ndimage import gaussian_filter
import matplotlib.colors as color
import time
plt.rcParams.update({'font.weight': 'bold'})
plt.rc('axes', titlesize=15)  # fontsize of the axes titles
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

def get_sat_info(sat_bias_current, sat_gen_freq, sat_gen_power, sat_vna_freq, sat_vna_powers, sat_gain, levels = [-2,-1.5,-1, -0.25, 0.25,1, 1.5,2], norm_power = -40, x_val = None, filter_window = 0, vmin = -1, vmax = 1, plot = True, xlim = None, ylim = None): 
    y_norm_val = max([norm_power, np.min(sat_vna_powers)+2]) #Signal power at which to normalize the rest of the plot to
    # print(f'Normalizing saturation to {y_norm_val} VNA power')
    # print(f'lowest VNA power: {np.min(sat_vna_powers)}')
    # print(f'filter_window: {filter_window}')
    # print(f'Size of gen freqs: {np.size(sat_gen_freq)}')
    # print(f'Size of gen pows: {np.size(sat_gen_power)}')
    # print(f'shape of saturation data: {np.shape(sat_gain)}')
    
    zero_freq_loc = np.argmin(sat_gen_power)
    zero_freq = sat_gen_freq[zero_freq_loc]
    
    if plot: 
        fig, ax, img = make_sat_img_plot(sat_bias_current, sat_vna_freq, sat_vna_powers, sat_gain, norm_power = y_norm_val, levels = levels, filter_window = filter_window, vmin = vmin, vmax = vmax)
        ax.set_title(sat_bias_current)
        cb = fig.colorbar(img, ax = ax)
        ax.set_xlabel('Generator Frequency(GHz)')
        ax.set_ylabel('Signal Power (dBm)')
        cb.set_label("S21 change from 20dB (dB)")

    smoothed_normed_data = log_normalize_to_row(sat_gen_freq, sat_vna_powers[0], gaussian_filter(sat_gain.T, (filter_window,0)), y_norm_val= y_norm_val)
    #getting saturation points
    sat_powers = []
    sat_gen_powers = []
    detunings = []
    sat_center_freqs = []
    
    for i, col in enumerate(smoothed_normed_data.T): 
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

        loc_arr = np.array([pos_loc, neg_loc])
        loc_arr = np.floor(loc_arr[np.logical_not(np.isnan(loc_arr))]).astype(int)

        loc = np.min(loc_arr)
        sat_powers.append(sat_vna_powers[0][loc])
        # print(sat_gen_power)
        sat_gen_powers.append(sat_gen_power[i])
        
        detunings.append(sat_gen_freq[i]-zero_freq)
        sat_center_freqs.append(sat_gen_freq[i])
        
    max_loc = np.where(sat_powers[0:-1] == np.max(sat_powers[0:-1]))[0][0]
    
    return (np.array(sat_gen_freq+(sat_gen_freq[1]-sat_gen_freq[0])/2)/1e6)[max_loc], sat_vna_freq[max_loc], sat_powers[max_loc], np.array(sat_powers), np.array(sat_gen_powers), np.array(detunings), np.array(sat_center_freqs), sat_bias_current*np.ones(len(detunings))

def superSat(filepaths, 
             y_norm_val = -70, 
             filter_window = 0, 
             conv = False, 
             conv_func = None, 
             tla_signal = 0, 
             tla_pump = 0, 
             plot_individual = False, 
             cscale = 10e6, 
             kerr = False, 
             kerr_null: float = None, 
             kerr_scale: float = None, 
             scale_flip = False, 
             device_name: str = '', 
             quanta_flip = False, 
             xlim = None, 
             ylim = None): 
    '''
    assemble all of the saturation sweeps, extract the best (highest) 
    saturation power in (gen_freqs, gen_powers) space, plot vs current
    '''
    # plt.rcParams.update({'font.weight': 'bold'})
    # plt.rc('axes', titlesize=15)  # fontsize of the axes titles
    # plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    
    filenames = []
    
    sat_powers = []
    sat_gen_powers = []
    sat_detunings = []
    sat_bias_arr = []
    sat_center_freqs_arr = []
    
    best_sat_powers = []
    best_sat_gen_frequencies = []
    best_sat_gen_powers = []
    
    best_sat_vna_frequencies = []
    plot_currents = []
    
    for sat_filepath in filepaths: 
        #extract the best gen powers
        sat_dicts = all_datadicts_from_hdf5(sat_filepath)
        satDict = sat_dicts['data']
        sat_data = satDict.extract('sat_gain')
        [bias_current, sat_gen_freq, sat_gen_power, sat_vna_powers, sat_gain, sat_vna_freq] = [sat_data.data_vals('sat_bias_current'), 
                                                                                    sat_data.data_vals('sat_gen_freq'),
                                                                                    sat_data.data_vals('sat_gen_power'), 
                                                                                    sat_data.data_vals('sat_vna_powers'), 
                                                                                    sat_data.data_vals('sat_gain'), 
                                                                                    sat_data.data_vals('sat_vna_freq')
                                                                                    ]
        for current in np.unique(bias_current): #could be multiple bias currents in one single TACO datafile
            plot_currents.append(current)
            bp1 = bias_current == current
            # make_sat_contour_plot(b1_val, sf1, svp1, sg1, norm_power = -60, levels = [-2, -1, -0.05,0.05, 1, 2], x_val = 10.687e3)
            sf1, sgp1, svp1, sgp1, sg1, svf1 = sat_gen_freq[bp1], sat_gen_power[bp1], sat_vna_powers[bp1], sat_gen_power[bp1], sat_gain[bp1], sat_vna_freq[bp1]
            
            best_sat_gen_freq, best_sat_vna_freq, best_sat_pow, sat_power_arr, sat_gen_power_arr, sat_detuning_arr, sat_center_freqs, bias_arr = get_sat_info(
                current, 
                sf1, 
                sgp1, 
                svf1, 
                svp1, 
                sg1, 
                norm_power = y_norm_val, 
                levels = [-20, -1,1, 20], 
                x_val = None, 
                filter_window = filter_window, 
                vmin = -1.7, vmax = 1.7, 
                plot = plot_individual)
            
            best_sat_gen_frequencies.append(best_sat_gen_freq)
            best_sat_vna_frequencies.append(best_sat_vna_freq)
            best_sat_powers.append(best_sat_pow)
            
            sat_powers.extend(sat_power_arr)
            sat_gen_powers.extend(sat_gen_power_arr)
            sat_detunings.extend(sat_detuning_arr)
            sat_bias_arr.extend(bias_arr)
            sat_center_freqs_arr.extend(sat_center_freqs)
            
    # print(list(zip(plot_currents, best_sat_powers, best_sat_frequencies)))
    

    if conv: #requires fluxsweep_fitting file to already have been run in the same kernel
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt_powers = np.array(best_sat_powers)-tla_signal
        # print(conv)
        if quanta_flip:     
            p1 = ax1.plot(1-conv_func(np.array(plot_currents)), plt_powers, 'b.', markersize = 15)
        else: 
            p1 = ax1.plot(conv_func(np.array(plot_currents)), plt_powers, 'b.', markersize = 15)
        ax1.title.set_text(f'{device_name}Best Saturation Power vs. Flux')
        ax1.set_xlabel('Flux Quanta ($\Phi/\Phi_0$)')
        ax1.set_ylabel('Saturation Power (dBm)')
        if ylim is not None: 
            ax1.set_ylim(ylim[0], ylim[1])
        plt.grid()
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        colors2 = conv_func(np.array(sat_bias_arr))
        print(cscale)
        img = ax2.scatter(np.array(sat_gen_powers)-tla_pump, np.array(sat_powers)-tla_signal, c = colors2, cmap = 'viridis', vmin = np.min(colors2), vmax = np.max(colors2), zorder = 1)
        cb2 = fig2.colorbar(img, ax = ax2)
        cb2.set_label("Bias Flux ($\Phi/\Phi_0$)")
        ax2.title.set_text(f'{device_name}Saturation Power vs. Pump power')
        ax2.set_xlabel('Generator Power (dBm Cryo)')
        ax2.set_ylabel('Saturation Power (dBm Cryo)')
        if xlim is not None:
            ax2.set_xlim(xlim[0], xlim[1])
        if ylim is not None: 
            ax2.set_ylim(xlim[0], xlim[1])
        plt.grid()
        
        if kerr: 
            colors_arr = [color.hex2color('#000066'),color.hex2color('#444488'),color.hex2color('#4444FF'), color.hex2color('#03fc41'),color.hex2color('#03fc41'), color.hex2color('#FF4444'), color.hex2color('#884444'), color.hex2color('#660000')]
            tick_labels = ['- 1', '0', '+ 1']
            if scale_flip: 
                colors_arr.reverse()
                tick_labels.reverse()
            _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors_arr)
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            # print(conv)
            cscale /= 1e6
            scale = cscale
            img = ax1.scatter(np.array(sat_gen_powers)-tla_pump, np.array(sat_powers)-tla_signal, c = colors2, cmap = _cmap, vmin = kerr_null-kerr_scale, vmax = kerr_null+kerr_scale, zorder = 1)
            cb1 = fig1.colorbar(img, ax = ax1, ticks=[kerr_null-kerr_scale, kerr_null, kerr_null+kerr_scale])
            cb1.set_label("Self-Kerr (arb. units)")
            cb1.ax.set_yticklabels(tick_labels)
            ax1.title.set_text(f'{device_name}Saturation Power vs. Pump power')
            ax1.set_xlabel('Generator Power (dBm Cryo)')
            ax1.set_ylabel('Saturation Power (dBm Cryo)')
            if xlim is not None: 
                ax1.set_xlim(xlim[0], xlim[1])
            if ylim is not None: 
                ax1.set_ylim(ylim[0], ylim[1])
            plt.grid()
            
            
        #plot best saturation wrt signal frequency. ie for all vna frequencies, plot the best saturation. Generally this is the generator freq/2
        
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        img = ax3.scatter(np.array(sat_center_freqs_arr)/1e9, sat_gen_powers)
        ax3.set_xlabel("Signal Frequency (GHz)")
        ax3.set_ylabel("Saturation Power (dBm Cryo)")
        
        colors2 = np.array(sat_bias_arr)*1e6
        img = ax2.scatter(np.array(sat_gen_powers)-tla_pump, np.array(sat_powers)-tla_signal, c = colors2, cmap = 'magma', vmin = np.min(colors2), vmax = np.max(colors2), zorder = 1)
        cb2 = fig2.colorbar(img, ax = ax3)
        cb2.set_label("Bias Current ($\mu A$)")
        
        # ax1.vlines(conv_func(-0.173e-3), np.min(plt_powers), np.max(plt_powers), linestyles = 'dashed', colors = ['red'])
        
        # plt.figure(2)
        # p1 = plt.plot(conv_func(np.array(plot_currents)), np.array(best_sat_gen_frequencies)/1000, 'b.', markersize = 15, label = 'Generator Frequencies')
        # plt.plot(conv_func(currents)[filt], 2*res_freqs[filt]/1e9, 'r-', markersize = 5, label = '2x SNAIL Frequency')
        # plt.xlabel('Flux Quanta ($\Phi/\Phi_0)$')
        # plt.ylabel('Generator Frequency (GHz)')
        # plt.title('Generator Frequency at Best Saturation Point vs. Flux')
        # plt.vlines(conv_func(-0.173e-3), np.min(np.array(best_sat_gen_frequencies)/1000), np.max(np.array(best_sat_gen_frequencies)/1000), linestyles = 'dashed', colors = ['red'])
        # plt.grid()
        # plt.legend()
        
        # plt.figure(3)
        # p0 = plt.plot(conv_func(np.array(plot_currents)), np.array(best_sat_vna_frequencies)/1e9, 'b.', markersize = 15, label = 'Signal Frequencies')
        # p1 = plt.plot(conv_func(currents)[filt], res_freqs[filt]/1e9, 'r-', markersize = 5, label = 'SNAIL Frequency')
        # plt.xlabel('Flux Quanta ($\Phi/\Phi_0)$')
        # plt.ylabel('VNA CW Frequency (GHz)')
        # plt.title('Signal Frequency at Best Saturation Point vs. Flux')
        # plt.vlines(conv_func(-0.173e-3), np.min(np.array(best_sat_vna_frequencies)/1e9), np.max(np.array(best_sat_vna_frequencies)/1e9), linestyles = 'dashed', colors = ['red'])
        # plt.grid()
        # plt.legend()
        
        # plt.figure(4)
        # plt.plot(np.array(best_sat_vna_frequencies)/1e9, best_sat_powers, 'k.', label = 'VNA Frequencies', markersize = 15)
        # plt.plot(np.array(best_sat_gen_frequencies)/2000, best_sat_powers, 'b.', label = 'Generator Frequencies/2', markersize = 15)
        # plt.ylabel('Saturation Power (dBm)')
        # plt.xlabel('Generator/VNA Frequency (GHz)')
        # plt.title("Best Saturation Power vs. Signal Frequency")
        # plt.grid()
        # plt.legend()
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # print(conv)
    scale = cscale
    
    colors_arr = [color.hex2color('#000066'),color.hex2color('#444488'),color.hex2color('#4444FF'), color.hex2color('#03fc41'),color.hex2color('#03fc41'), color.hex2color('#FF4444'), color.hex2color('#884444'), color.hex2color('#660000')]
    if scale_flip: 
        colors_arr.reverse()
        tick_labels.reverse()
    _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors_arr)
    
    img = ax1.scatter(np.array(sat_gen_powers)-tla_pump, np.array(sat_powers)-tla_signal, c = np.array(sat_detunings)/1e6, cmap = _cmap, vmin = -cscale, vmax = cscale, zorder = 1)
    cb1 = fig1.colorbar(img, ax = ax1)
    cb1.set_label("Detuning (MHz)")
    ax1.title.set_text(f'{device_name}Saturation Power vs. Pump power')
    ax1.set_xlabel('Generator Power (dBm Cryo)')
    ax1.set_ylabel('Saturation Power (dBm Cryo)')
    if xlim is not None: 
        ax1.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])
    plt.grid()
    fig1.savefig(r'G:\My Drive\old files\WRITE - Presentations\mpl_figures\test.pdf', format = 'pdf')
    
    
    
    
def superTACO_Bars(filepaths, angles = [45,45], quanta_size = None, quanta_offset = None, bardims = [1,1], barbase = -30, plot = False):
    #step 1: assemble best powers into bias_currents vs. (gen_freq vs. best_powers) array
    #ie for each n bias current there is a gen_freq array
    #and for each m(n) gen_freq there is one gen_power that is best (could be NaN if garbage)
    #feed this into n mplot3d commands each with their own oclor and legend label

    
    bias_currents = []
    best_gen_frequencies = []
    best_gen_powers = []
    gains = []
    info_dict = {}
    
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
            print(f"CURRENT: {current*1000}mA")
            filt = bias_current == current
            cfreqs = gen_frequency[filt]
            cpowers = gen_power[filt]
            unique_freqs = np.unique(cfreqs)
            cgain = calc_gain[filt]
            
            best_powers = [select_closest_to_target(cpowers[cfreqs == f], cgain[cfreqs == f], 20) for f in unique_freqs]
            best_power = np.min(best_powers)
            best_gen_powers.append(best_power)
            
            best_gains = [select_closest_to_target(cgain[cfreqs == f], cgain[cfreqs == f], 20) for f in unique_freqs]
            best_gain = best_gains[np.argmin(np.abs(np.array(best_gains)-20))]
            gains.append(best_gain)

            #convert freqs to detuning from best power
            best_freq = np.average(unique_freqs[np.where(best_powers == best_power)])
            best_gen_frequencies.append(best_freq)
            
            
            adjusted_freqs = unique_freqs - best_freq
            
            info_dict[current] = adjusted_freqs, best_powers
    if plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.invert_yaxis()
        ax.azim = angles[0]
        ax.elev = angles[1]   
        #plotting in sorted order of bias currents for rendering reasons
        for i, current in enumerate(sorted(info_dict)): 
            # print(f"CURRENTS: {current}")
            adjusted_freqs = info_dict[current][0]
            best_powers = info_dict[current][1]
            # print(f"FREQS: {adjusted_freqs}")
            # print(f"POWERS: {best_powers}")
            if quanta_size != None:
                quant_frac = np.round((current-quanta_offset)/quanta_size, 3)
                base_of_bars = barbase*np.ones(np.size(best_powers))
                height_of_bars = best_powers-base_of_bars
                # print(list(zip(base_of_bars, height_of_bars)))
                ax.bar3d(current*np.ones(np.size(adjusted_freqs))*1000, adjusted_freqs/1e6, base_of_bars , bardims[0], bardims[1], height_of_bars, color=None, shade=True, label = f'{quant_frac} quanta', zsort = 'average')
    
        ax.set_xlabel("Bias Current (mA)")
        ax.set_ylabel("Generator Detuning (MHz)")
        ax.set_zlabel("Generator Power (dBm)")
        ax.set_title("20dB Gain Power vs. Flux Bias and Generator Detuning")
    
    return [info_dict, np.array(bias_currents), np.array(best_gen_frequencies), np.array(best_gen_powers), np.array(gains)]

#%%
if __name__ == "__main__": 
    # total_line_attenuation_signal = 0
    # total_line_attenuation_pump = 0 #does not include VNA attenuation
    # gain_cwd = r'Z:\Data\SA_1X_C1\Best_Tacos\Gain'
    # res = find_all_ddh5(gain_cwd)
    # [info_dict, bias_currents, best_gen_freqs, best_gen_powers, gains] = superTACO_Bars(res, angles = [60,20], quanta_size = 0.35e-3, quanta_offset = -0.071e-3, bardims = [0.001, 0.7], barbase = -24)
    
    # best_gen_powers -= total_line_attenuation_pump
    
    # fig2 = plt.figure(2)
    # conv = False
    # if conv: 
    #     # plt.plot(conv_func(bias_currents), np.array(best_gen_powers)-total_line_attenuation_signal, 'b.', markersize = 15)
    #     plt.title(r'Lowest 20dB Power (dBm) vs. Flux ($\Phi_0$)')
    #     plt.xlabel('Flux Quanta ($\Phi/\Phi_0)$')
    #     plt.ylabel('Generator Power @20dB Gain (dBm)')
    #     plt.grid()
    #     # plt.vlines(conv_func(-0.173e-3), np.min(best_gen_powers-total_line_attenuation_pump), np.max(best_gen_powers-total_line_attenuation_pump), linestyles = 'dashed', colors = ['red'])
    # else: 
    #     plt.plot(bias_currents*1000, best_gen_powers, 'b.', markersize = 15)
    #     plt.title('Lowest 20dB Power (dBm RT) vs. Bias Current (mA)')
    #     plt.xlabel('Bias Current (mA)')
    #     plt.ylabel('Generator Power @20dB Gain (dBm)')
    #     plt.grid()
    
    
    sat_cwd = r'Z:\Data\SA_2X_B1\best_tacos\sat'
    res = find_all_ddh5(sat_cwd)
    superSat(res, y_norm_val = -92, filter_window=9, conv = False, plot_individual = False, tla_signal = 61, tla_pump = 71.5, cscale = 100)

