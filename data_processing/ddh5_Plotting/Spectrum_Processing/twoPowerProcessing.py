# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 00:17:40 2021

@author: Hatlab-RRK
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plottr.data.datadict_storage import all_datadicts_from_hdf5

#goal: take a 2-power sweep and extract the behavior at certain frequencies
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp4/trace/TwoPowerSpec/2021-10-21/2021-10-21_0004_TPS_bp4_more_avgs/2021-10-21_0004_TPS_bp4_more_avgs.ddh5'

specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('Spectrum_power')['Spectrum_frequency']['values']
sig_powers = specData.extract('Spectrum_power')['Signal_power']['values']
pump_powers = specData.extract('Spectrum_power')['Pump_power']['values']+60
spec_powers = specData.extract('Spectrum_power')['Spectrum_power']['values']

#%% make image plots of the spectrum power vs gen and pump powers at some given spectrum frequency
plt_freqs = [6.8e9-300e3, 6.8e9-100e3, 6.8e9, 6.8e9+100e3, 6.8e9+300e3]
for spec_freq in plt_freqs: 
    spec_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs-spec_freq)))]
    filt = spec_freqs == spec_freq
    
    fig, ax = plt.subplots()
    img = ax.tricontourf(sig_powers[filt],pump_powers[filt], spec_powers[filt], levels = np.arange(-80, -30+1, 3), cmap = 'magma')
    ax.set_xlabel("Signal power (dBm)")
    ax.set_ylabel("Pump power (dBm)")
    ax.set_title(f"PSD at f = {np.round(spec_freq/1e9, 4)} GHz")
    cb = plt.colorbar(img)
    cb.set_label("Power Spectral Density (dBm)")
    
#%%plot the difference of two of them
f1 = 6.8e9-700e3 
f2 = 6.8e9+700e3

spec_freq1 = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs-f1)))]
spec_freq2 = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs-f2)))]
filt1 = spec_freqs == spec_freq1
filt2 = spec_freqs == spec_freq2

fig, ax = plt.subplots()
img = ax.tricontourf(sig_powers[filt1],pump_powers[filt1], spec_powers[filt1]-spec_powers[filt2], levels = np.arange(-10, 10+1, 1), cmap = 'seismic')
ax.set_xlabel("Signal power (dBm)")
ax.set_ylabel("Pump power (dBm)")
ax.set_title(f"{np.round(spec_freq1/1e9, 4)} GHz - {np.round(spec_freq2/1e9, 4)} GHz")
cb = plt.colorbar(img)
cb.set_label("Power Spectral Density (dBm)")
#%%extract spec analyzer trace at a combo of signal and pump powers
sig_power = -67
pump_power = 2.337
sig_power = np.unique(sig_powers)[np.argmin(np.abs(np.unique(sig_powers-sig_power)))]
pump_power = np.unique(pump_powers)[np.argmin(np.abs(np.unique(pump_powers-pump_power)))]
filt = (sig_powers == sig_power)*(pump_powers == pump_power)
fig, ax = plt.subplots()
ax.plot((spec_freqs[filt]-6.8e9)/1e3, spec_powers[filt])
ax.set_xlabel(r"$f-\frac{\omega_0}{2\pi}$  (kHz)")
ax.set_ylabel("PSD (dBm)")
ax.grid()



#%% make image plots of the spectrum power vs gen and pump powers at some given spectrum frequency
plt_freqs = [6.8e9-300e3, 6.8e9-100e3, 6.8e9, 6.8e9+100e3, 6.8e9+300e3]
freq_to_label = {6.7997:'$\omega_0-3\Delta$', 
                6.7999:'$\omega_0-\Delta$', 
                6.8000:'$\omega_0$', 
                6.8001:'$\omega_0+\Delta$', 
                6.8003:'$\omega_0+3\Delta$'
                }
for plt_power in np.unique(pump_powers): 
    # plt_power = -56.8
    # plt_power = -58.3
    # plt_power = np.max(pump_powers)
    # plt_power = -57.7
    fig, ax = plt.subplots()
    for spec_freq in plt_freqs: 
        spec_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs-spec_freq)))]
        pump_power = np.unique(pump_powers)[np.argmin(np.abs(np.unique(pump_powers-plt_power)))]
        filt = (spec_freqs == spec_freq)*(pump_powers == pump_power)
        
        ax.plot(sig_powers[filt], spec_powers[filt], label = f'{freq_to_label[np.round(spec_freq/1e9, 4)]}')
        ax.set_xlabel("Signal power (dBm)")
        ax.set_ylabel("Power Spectral Density (dBm)")
        ax.set_title(f"Pump power = {np.round(plt_power, 3)} dBm")
        ax.legend()
        ax.grid()