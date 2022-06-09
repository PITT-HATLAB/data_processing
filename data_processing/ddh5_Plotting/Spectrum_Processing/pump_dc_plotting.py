# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 00:17:40 2021

@author: Hatlab-RRK
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plottr.data.datadict_storage import all_datadicts_from_hdf5


#goal: take a power sweep and extract the behavior at certain frequencies to plot in a more functional way
filepath = r'Z:/Data/N25_L3_SP_2/time-domain/30dB_4MHz/SA_power_sweep/2022-06-01_0002_pow_sweep_VNA_signal_src_SA_SA.ddh5'
filepath = r'Z:/Data/N25_L3_SP_2/time-domain/30dB_4MHz/SA_power_sweep/2022-06-01_0003_pow_sweep_VNA_signal_src_SA_SA.ddh5'
filepath = r'Z:/Data/N25_L3_SP_2/time-domain/loopback_pwr_sweep/spectra_file/2022-06-03_0002_spectra.ddh5'
# filepath = r'Z:/Data/N25_L3_SP_2/time-domain/loopback_pwr_sweep_no_gain_block/spectra_file/2022-06-03_0001_spectra.ddh5'
# filepath = r'Z:/Data/N25_L3_SP_2/time-domain/loopback_pwr_sweep_signal_disconnected/spectra_file/2022-06-03_0001_spectra.ddh5'
filepath = r'Z:/Data/N25_L3_SP_2/time-domain/loopback_pwr_sweep_1_gain_block/spectra_file/2022-06-03_0001_spectra.ddh5'

filepath = r'Z:/Data/N25_L3_SP_2/time-domain/through_amp_0dB_att/spectra_file/2022-06-03_0001_spectra.ddh5'

filepath = r'Z:/Data/N25_L3_SP_2/time-domain/pump_downconv_sweep/SA_pwr_sweep/2022-06-06_0001_pow_sweep_VNA_signal_src_SA_SA.ddh5'

ind_pwr_name = 'Signal Power (dBm RT)'
dep_pwr_name = 'spec_power'
freq_name = 'spec_frequency'

# ind_pwr_name = 'ind_var'
# dep_pwr_name = 'power'
# freq_name = 'frequency'

specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract(dep_pwr_name)[freq_name]['values']
spec_powers = specData.extract(dep_pwr_name)[dep_pwr_name]['values']
gen_powers = specData.extract(dep_pwr_name)[ind_pwr_name]['values']
#take middle value for LO leakage
# detuning = 15
# cf = 50
detuning = 4e6
sf = 5601900000.0+2e6

pump_dc_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(5601900000.0)))]
pump_signal_imd = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(5601900000.0+detuning)))]
pump_idler_imd = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(5601900000.0-detuning)))]

signal_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(sf)))]
idler_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(signal_freq-detuning)))]

idler_IMD1_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(signal_freq-2*detuning)))]
signal_IMD1_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(signal_freq+detuning)))]

idler_IMD2_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(signal_freq-3*detuning)))]
signal_IMD2_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(signal_freq+2*detuning)))]

pump_filt = spec_freqs == pump_dc_freq
pump_signal_IMD_filt = spec_freqs == pump_signal_imd
pump_idler_IMD_filt = spec_freqs == pump_idler_imd


signal_filt = spec_freqs == signal_freq
idler_filt = spec_freqs == idler_freq

signal_IMD1_filt = spec_freqs == signal_IMD1_freq
idler_IMD1_filt = spec_freqs == idler_IMD1_freq

signal_IMD2_filt = spec_freqs == signal_IMD2_freq
idler_IMD2_filt = spec_freqs == idler_IMD2_freq

log = 1
if log: 
    plt_powers = gen_powers
else: 
    plt_powers = 20*np.log10(gen_powers)

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(plt_powers[pump_filt], spec_powers[pump_filt], '.', label = 'pump downconversion power (dBm)')
ax.plot(plt_powers[pump_signal_IMD_filt], spec_powers[pump_signal_IMD_filt], '.', label = 'pump+idler_IMD power (dBm)')
ax.plot(plt_powers[pump_idler_IMD_filt], spec_powers[pump_signal_IMD_filt], '.', label = 'pump+signal_IMD (dBm)')

ax.plot(plt_powers[signal_filt], spec_powers[signal_filt], '.', label = 'Signal Power (dBm)')
ax.plot(plt_powers[idler_filt], spec_powers[idler_filt], '.', label = 'Idler Power (dBm)')
ax.plot(plt_powers[signal_IMD1_filt], spec_powers[signal_IMD1_filt], '.', label = 'Signal IMD1 power (dBm)')
ax.plot(plt_powers[idler_IMD1_filt], spec_powers[idler_IMD1_filt], '.', label = 'idler IMD1 power (dBm)')
ax.plot(plt_powers[signal_IMD2_filt], spec_powers[signal_IMD2_filt], '.', label = 'Signal IMD2 power (dBm)')
ax.plot(plt_powers[idler_IMD2_filt], spec_powers[idler_IMD2_filt], '.', label = 'idler IMD2 power (dBm)')

ax.set_xlabel('Signal generator power (dBm)')
ax.legend(bbox_to_anchor = (1,1))
ax.grid()
ax.set_ylabel('Spectrum Power (dBm)') 
ax.set_title(f'IMD plot ')
ax.set_aspect(0.5)

fig, ax = plt.subplots(figsize = (8,6))
ax.plot(plt_powers[pump_filt], np.diff(spec_powers[pump_filt], prepend = 0), '.', label = 'pump downconversion power (dBm)')
ax.plot(plt_powers[pump_signal_IMD_filt], np.diff(spec_powers[pump_signal_IMD_filt], prepend = 0), '.', label = 'pump+idler_IMD power (dBm)')
ax.plot(plt_powers[pump_idler_IMD_filt], np.diff(spec_powers[pump_signal_IMD_filt], prepend = 0), '.', label = 'pump+signal_IMD (dBm)')

ax.plot(plt_powers[signal_filt], np.diff(spec_powers[signal_filt], prepend = 0), '.', label = 'Signal Power (dBm)')
ax.plot(plt_powers[idler_filt], np.diff(spec_powers[idler_filt], prepend = 0), '.', label = 'Idler Power (dBm)')
ax.plot(plt_powers[signal_IMD1_filt], np.diff(spec_powers[signal_IMD1_filt], prepend = 0), '.', label = 'Signal IMD1 power (dBm)')
ax.plot(plt_powers[idler_IMD1_filt], np.diff(spec_powers[idler_IMD1_filt], prepend = 0), '.', label = 'idler IMD1 power (dBm)')
ax.plot(plt_powers[signal_IMD2_filt], np.diff(spec_powers[signal_IMD2_filt], prepend = 0), '.', label = 'Signal IMD2 power (dBm)')
ax.plot(plt_powers[idler_IMD2_filt], np.diff(spec_powers[idler_IMD2_filt], prepend = 0), '.', label = 'idler IMD2 power (dBm)')

ax.set_xlabel('Signal generator power (dBm)')
ax.legend(bbox_to_anchor = (1,1))
ax.grid()
ax.set_ylabel('Spectrum Power (dBm)') 
ax.set_title(f'IMD plot ')
ax.set_aspect(0.5)
ax.set_ylim(-10, 10)
ax.hlines([1,2,3], -20, 0, color = 'k', linestyle = '-.')

plt.figure()
fsize = np.size(np.unique(spec_freqs))
fwindow = [sf-detuning*5, sf+detuning*5]
[fw0, fw1] = fwindow

plt_power = gen_powers[np.argmin(np.abs(np.unique(gen_powers)))]
gen_filt = gen_powers == -10
#end: 
# plt_freqs = spec_freqs[-fsize:]

#beginning: 
plt_freqs = spec_freqs[gen_filt]

ffilt = (plt_freqs>=fw0)*(plt_freqs<fw1)
# plt.plot(spec_freqs[0: fsize], spec_powers[0: fsize])
plt_powers =  spec_powers[gen_filt]
plt.plot(plt_freqs[ffilt],plt_powers[ffilt])
plt.vlines([pump_dc_freq, pump_signal_imd, pump_idler_imd,
            signal_freq, signal_IMD1_freq, #signal_IMD2_freq, 
            idler_freq, idler_IMD1_freq, #idler_IMD2_freq,
            ], -80, 20, color = ['green', 'green', 'green', 'blue','blue', 'orange', 'orange'], linestyle = '--')



