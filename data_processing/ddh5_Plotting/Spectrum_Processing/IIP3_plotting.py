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
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/IIP3_SA_sweep/2021-09-13/2021-09-13_0004_6.064GHz_20dB_Gain_IIP3_test/2021-09-13_0004_6.064GHz_20dB_Gain_IIP3_test.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/IIP3_SA_sweep/2021-09-13/2021-09-13_0005_6.064GHz_20dB_Gain_IIP3_test/2021-09-13_0005_6.064GHz_20dB_Gain_IIP3_test.ddh5'

specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('power')['CXA_frequency']['values']
spec_powers = specData.extract('power')['power']['values']
gen_powers = specData.extract('power')['Gen_power']['values']
#take middle value for LO leakage
detuning = 1e3
center_freq = np.unique(spec_freqs)[np.size(np.unique(spec_freqs))//2]
lower_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-detuning)))]
IM_spur_lower = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-3*detuning)))]

upper_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+detuning)))]
IM_spur_upper = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+3*detuning)))]


lower_sideband_filt = spec_freqs == lower_sideband_freq
upper_sideband_filt = spec_freqs == upper_sideband_freq
IM_spur_lower_filt = spec_freqs == IM_spur_lower
IM_spur_upper_filt = spec_freqs == IM_spur_upper

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
# ax.plot(gen_powers[leakage_filt], spec_powers[leakage_filt], label = 'LO leakage (dBm)')
ax.plot(gen_powers[upper_sideband_filt], spec_powers[upper_sideband_filt], label = 'Upper input tone power (dBm)')
ax.plot(gen_powers[lower_sideband_filt], spec_powers[lower_sideband_filt], label = 'Lower input tone power (dBm)')
ax.plot(gen_powers[IM_spur_upper_filt], spec_powers[IM_spur_upper_filt], label = 'Upper spur power (dBm)')
ax.plot(gen_powers[IM_spur_lower_filt], spec_powers[IM_spur_lower_filt], label = 'Lower spur power (dBm)')

ax.set_xlabel('LO generator power (dBm)')
ax.legend()
ax.grid()
ax.set_ylabel('Spectrum Power (dBm)') 
ax.set_title(f'Amplifier Low-power gain: 20dB, f1-f2: {np.round(detuning/1e3)} kHz ')

