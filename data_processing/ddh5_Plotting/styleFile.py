# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:47:55 2022

@author: Hatlab-RRK

creating a matplotlib style file
"""

from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np


mpl_dir = 'C:\\Users\\Hatlab-RRK\\.matplotlib'

plt.style.use('hatlab')

#%% testing data

filepath = r'Z:/Data/N25_L3_SQ/traces/gain/2022-04-25/2022-04-25_0001_bp1_gain/2022-04-25_0001_bp1_gain.ddh5'

dd = all_datadicts_from_hdf5(filepath)['data']
pows = dd.extract('power')['power']['values']
freqs = dd.extract('power')['frequency']['values']

#plot the LO leakage vs power
fig, ax = plt.subplots()
ax.plot(freqs/1e9, pows)
ax.set_xlabel('VNA frequency (GHz)')

ax.set_ylabel('VNA Gain (dB)')
ax.legend()
ax.grid()
ax.set_title(f'0.06mA, {8.65-30-10}dBm Cryo, +277kHZ generator detuning: 6.6MHz BW')