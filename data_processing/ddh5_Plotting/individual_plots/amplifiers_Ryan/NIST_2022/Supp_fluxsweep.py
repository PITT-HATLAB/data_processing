# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:05:02 2023

@author: Ryan
"""

goal_f0 = 7.5e9
goal_g_sss = 20e6
goal_p = 0.35
goal_kappa = 70e6
from data_processing.models.snailamp import SnailAmp, find_quanta
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from ipywidgets import widgets
from scipy.interpolate import interp1d
import proplot as pplt

from plottr.data import datadict_storage as dds

#Fluxsweep and processing - Late August 2021
from data_processing.ddh5_Plotting.utility_modules.FS_utility_functions import fit_fluxsweep
#for N25L3SP_3
datadir = r'C:\Users\Ryan\OneDrive - University of Pittsburgh\paper_data\NISTAMP_2022\fluxsweep\2022-06-17_0002_FS_-40dBm_VNA_VNA.ddh5'
fitdir = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/fluxsweep/2022-06-17_0001_N25_L3_SP.ddh5'
#for N25L3SP_2
datadir2 = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/fluxsweep/N25L3SP2/2022-06-10_0003_FS_-60dBm_VNA_VNA.ddh5'
fitdir2 = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/fluxsweep/N25L3SP2/2022-07-26_0001_N25_L3_SP_2.ddh5'
# savedir = r'Z:\Data\N25_L3_SP_3\fluxsweeps\fits'
# FS = fit_fluxsweep(datadir, savedir, device_name, 
#                   phaseName = 'vna_phase', 
#                   currentName = 'bias_current', 
#                   powerName = 'vna_power', 
#                   freqName = 'vna_frequency')

# filt = FS.currents > FS.currents[0]
# currents, res_freqs, Qints, Qexts, magBacks = FS.semiauto_fit(FS.currents[filt], 
#                                                               FS.vna_freqs[filt]/(2*np.pi), 
#                                                               FS.undriven_vna_power[filt], 
#                                                               FS.undriven_vna_phase[filt], 
#                                                               FS.initial_popt, 
#                                                               debug = False, 
#                                                               savedata = True, 
#                                                               smooth = False, 
#                                                               smooth_win = 5, 
#                                                               adaptive_window = False, 
#                                                               adapt_win_size = 1000e6, 
#                                                               fourier_filter = False, 
#                                                               pconv_tol = 10)

#plot the fluxsweep in the color scheme that we're used to
data = dds.all_datadicts_from_hdf5(datadir)['data']
data2 = dds.all_datadicts_from_hdf5(datadir2)['data']

#%%pull info out of the fit file and plot on top of pcolor plot
fit_data = dds.all_datadicts_from_hdf5(fitdir)['data']
fit_data2 = dds.all_datadicts_from_hdf5(fitdir2)['data']
#%%plot the raw data
fig, axs = pplt.subplots(ncols = 2, nrows = 1)
for i, (fs_data, fs_fit_data, name, ax) in enumerate(zip([data, data2], [fit_data, fit_data2], ['N25L3SP3', 'N25L3SP2'], axs)):
    currents = np.unique(fs_data['bias_current']['values'])
    fmin = 5.25e9
    fmax = 6.05e9
    ffilt = (fs_data['vna_frequency']['values'] >= fmin)*(fs_data['vna_frequency']['values'] <= fmax)
    freqs = np.unique(fs_data['vna_frequency']['values'][ffilt])/1e9
    phase = fs_data['vna_phase']['values'][ffilt]
    
    fit_currents = fs_fit_data['current']['values']
    fit_freq = fs_fit_data['base_resonant_frequency']['values']
    fit_Qint = fs_fit_data['base_Qint']['values']
    fit_Qext = fs_fit_data['base_Qext']['values']
    
    # phase_vals = np.linspace(-np.pi, np.pi, 1000)
    
    vmin = -np.pi
    vmax = np.pi
    im = ax.pcolormesh(currents*1000, freqs, phase.reshape((np.size(currents), np.size(freqs))).T, discrete = False, cmap = 'seismic_r', vmin = vmin, vmax = vmax)
    
    
    ax.plot(fit_currents*1000, fit_freq/1e9,'k--')
    # ax.plot(currents2*1000, res_freqs2/1e9, label = 'cooldown_2')
    ax.set_xlabel('Bias Current (mA)')
    ax.set_ylabel('Frequency (GHz)')
    ax.title.set_text(name+' Flux Biasing')
    ax.set_ylim(fmin/1e9, fmax/1e9)
    # ax.legend()
    ax.grid()
fig.colorbar(im)
fig.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAmp_2022\raw_plots\\'+name+'_fs_plot.png')
    
    # print(f"{(np.max(res_freqs/1e9),np.min(res_freqs/1e9) )}")
    
    # #plot the kappa
    # from scipy.signal import savgol_filter
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(currents*1000, res_freqs*(1/Qexts+1/Qints)/1e6)
    # # ax.plot(currents2*1000, res_freqs2*(1/Qexts2+1/Qints2)/1e6)
    # ax.set_xlabel('Bias Currents (mA)')
    # ax.set_ylabel('Kappa (MHz)')
    # ax.title.set_text(device_name+' linewidth vs. Bias Current')
    # ax.grid()
