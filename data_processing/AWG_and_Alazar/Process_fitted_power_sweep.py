# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:46:39 2021

@author: Hatlab-RRK
"""
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

line_fit = lambda x, slope, offset: slope*x+offset

fitDir = r'Z:\Data\C1\C1_Hakan\Gain_pt_0.103mA\Pump_power_sweeps\1_fit\2021-07-01\2021-07-01_0001_LO_PWR_PHASE_fits\2021-07-01_0001_LO_PWR_PHASE_fits.ddh5'

datadicts = all_datadicts_from_hdf5(fitDir)['data']

phase = datadicts.extract('avg_power_gain')['phase']['values']
p_val = 0

avg_power_gain = datadicts.extract('avg_power_gain')['avg_power_gain']['values'][phase == p_val]

pump_power = datadicts.extract('avg_power_gain')['pump_power']['values'][phase == p_val]

sep_over_sigma = datadicts.extract('sep_over_sigma')['sep_over_sigma']['values'][phase == p_val]

#fit line and fit

fig, ax = plt.subplots(figsize = [6,4])
popt, pcov = curve_fit(line_fit, pump_power[0:6], avg_power_gain[0:6], [1, 0])
ax.plot(pump_power, avg_power_gain, '.')
ax.plot(pump_power, line_fit(pump_power, *popt), '--')
ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dBm}$', (-9,20), fontsize = 20)
ax.grid()
ax.set_xlabel('Pump Power (dBm)')
ax.set_ylabel('Power Gain (dB)')

fig, ax = plt.subplots(figsize = [6,4])
ax.plot(10**(avg_power_gain/20), sep_over_sigma, '.')
popt, pcov = curve_fit(line_fit, 10**(avg_power_gain/20)[0:8], sep_over_sigma[0:8], [1, 0])
ax.plot(10**(avg_power_gain/20), line_fit(10**(avg_power_gain/20), *popt), '--')
ax.annotate(f'Slope: {np.round(popt[0], 3)}', (4,4), fontsize = 20)
ax.grid()
ax.set_xlabel('Power_Gain (V/V)')
ax.set_ylabel('sep_over_sigma (mV/mV)')