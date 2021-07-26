# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:59:08 2021

@author: Hatlab_3
"""
from data_processing.ddh5_Plotting.utility_modules.FS_utility_functions import fit_fluxsweep
from data_processing.Helper_Functions import find_all_ddh5
from plottr.apps.autoplot import autoplotDDH5, script, main
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d

def find_quanta(currents, res_freqs, show = True, smooth_window = 11, order = 2): 
    ext = argrelextrema(savgol_filter(res_freqs, smooth_window, 2), np.greater, order = order)[0]
    if show: 
        plt.plot(currents, res_freqs)
        for pt in ext: 
            plt.plot(currents[pt], res_freqs[pt], 'r*')
    if np.size(ext) == 2: 
        quanta_size = np.abs(currents[ext[1]]-currents[ext[0]])
        quanta_offset = min(currents[ext])
    else: 
        raise Exception(f'Two extrema not found: {ext}')
    current_to_quanta_conversion_function = lambda c: (c-quanta_offset)/quanta_size
    quanta_to_current_function = lambda q: q/quanta_size+min(currents[ext], key=abs)
    return quanta_size, quanta_offset, current_to_quanta_conversion_function, quanta_to_current_function
if __name__ == '__main__': 
    
#adapting an old file to a new file
    #%%
    datadir = r'Z:/Data/SA_2X_B1/fluxsweep/2021-07-09/2021-07-09_0001_B1_FS1/2021-07-09_0001_B1_FS1.ddh5'
    savedir = r'Z:/Data/SA_2X_B1/fluxsweep/fits'
    # datadir = r'E:\Data\Cooldown_20210104\fluxsweep\2021-01-04_0003_Recentering_FS.ddh5'
    # savedir = r'E:\Data\Cooldown_20210104\fluxsweep'
    
    FS = fit_fluxsweep(datadir, savedir, 'SA_2X_B1')
    #%%
    FS.initial_fit(8.25e9, QextGuess = 1e2, QintGuess=20e4, magBackGuess = 0.01, phaseOffGuess = 0, debug = False, smooth = False, smooth_win = 15, adaptive_window = False, adapt_win_size = 100e6)
    #%% Automatic Fitting (be sure initial fit is good!)
    currents, res_freqs, Qints, Qexts, magBacks = FS.semiauto_fit(FS.currents, FS.vna_freqs/(2*np.pi), FS.undriven_vna_power, FS.undriven_vna_phase, FS.initial_popt, debug = False, savedata = True, smooth = False, smooth_win = 5, adaptive_window = True, adapt_win_size = 300e6, fourier_filter = False, pconv_tol = 7)
    #%%reloading an old file
    #%%plotting the resonant frequency
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(currents*1000, res_freqs/1e6)
    ax.set_xlabel('Bias Currents (mA)')
    ax.set_ylabel('Resonant Frequencies (MHz)')
    ax.title.set_text('ChemPot Resonant Frequency vs. Bias Current')
    #%%Finding and plotting flux quanta and flux variables, interpolating resonance frequencies to generate resonance functions wrt bias current and flux
    quanta_size, quanta_offset, conv_func, conv_func_inverse = find_quanta(currents, res_freqs, show = False, smooth_window = 221)
    res_func = interp1d(currents, res_freqs, 'linear')
    print(f"Quanta size: {quanta_size}\nQuanta_offset: {quanta_offset}")
    filt = (conv_func(currents)<0)*(conv_func(currents)>-0.52)
    plt.plot(conv_func(currents)[filt], res_freqs[filt])
    plt.figure(2)
    plt.plot(currents, res_freqs, label = 'fitted data')
    plt.plot(currents, res_func(currents), label = 'quadratic interpolation')
    plt.legend()
    plt.figure(3)
    #%%
    plt.plot(currents, res_func1(currents)-savgol_filter(res_func(currents), 21, 2))