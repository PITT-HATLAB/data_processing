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
filepath = r'Z:/Data/00_Calibrations/RT Equipment calibrations/XMW_interferometer_rev2_qubit_drive/2021-09-12/2021-09-12_0005_Qubit_drive_module_LO_pwr_vs_sideband_with_splitter_4/2021-09-12_0005_Qubit_drive_module_LO_pwr_vs_sideband_with_splitter_4.ddh5'

specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('power')['CXA_frequency']['values']
spec_powers = specData.extract('power')['power']['values']
gen_powers = specData.extract('power')['Gen_power']['values']
#take middle value for LO leakage
LO_leakage_freq = np.unique(spec_freqs)[np.size(np.unique(spec_freqs))//2]
lower_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(LO_leakage_freq-50e6)))]
upper_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(LO_leakage_freq+50e6)))]

leakage_filt = spec_freqs == LO_leakage_freq
lower_sideband_filt = spec_freqs == lower_sideband_freq
upper_sideband_filt = spec_freqs == upper_sideband_freq

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(gen_powers[leakage_filt], spec_powers[leakage_filt], label = 'LO leakage (dBm)')
ax.plot(gen_powers[upper_sideband_filt], spec_powers[upper_sideband_filt], label = 'Upper sideband power (dBm)')
ax.plot(gen_powers[lower_sideband_filt], spec_powers[lower_sideband_filt], label = 'Lower sideband power (dBm)')
ax.set_xlabel('LO generator power (dBm)')
ax.legend()
ax.grid()
ax.set_title('Tuned at 13dBm LO power')

