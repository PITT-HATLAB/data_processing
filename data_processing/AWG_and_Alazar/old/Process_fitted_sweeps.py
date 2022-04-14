# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:46:39 2021

@author: Hatlab-RRK
"""
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def avg_ind_vars(plot_var, plot_val):
    #independent values you want to keep to plot, dependent value you want to plot
    #will average over all others
    plot_val_final = []
    plot_var_final = []
    
    for plot_var_val in np.unique(plot_var): 
        filt = (plot_var == plot_var_val)
        
        plot_val_final.append(np.average(plot_val[filt]))
        plot_var_final.append(plot_var_val)
            
    # print(np.shape(plot_var_final), np.shape(plot_val_final))
    return np.array(plot_var_final), np.array(plot_val_final)

line_fit = lambda x, slope, offset: slope*x+offset

fitDir = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/Power_detuning_sweeps/1_fit/2021-07-07_0007_LO_VOLT_PHASE_fits/2021-07-07_0007_LO_VOLT_PHASE_fits.ddh5'
datadicts = all_datadicts_from_hdf5(fitDir)['data']

det_val = 1.5
p_val = 0

detuning = datadicts.extract('avg_power_gain')['detuning']['values']   
avg_power_gain = datadicts.extract('avg_power_gain')['avg_power_gain']['values']
fit_fidelity = datadicts.extract('fit_fidelity')['fit_fidelity']['values']
pump_power = datadicts.extract('avg_power_gain')['pump_pwr']['values']
sep_over_sigma = datadicts.extract('sep_over_sigma')['sep_over_sigma']['values']


phase = datadicts.extract('avg_power_gain')['phase']['values']

filt_detuning, filt_avg_power_gain = avg_ind_vars(detuning, avg_power_gain)

#fit line and fit

fig, ax = plt.subplots(figsize = [6,4])
# popt, pcov = curve_fit(line_fit, avg_ind_var(pump_power, phase), avg_ind_var(pump_power, phase), [1, 0])
ax.plot(filt_detuning, filt_avg_power_gain, '.')
# ax.plot(pump_power, line_fit(pump_power, *popt), '--')
# ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dB}$', (-9,20), fontsize = 20)
ax.set_xlabel('detuning (MHz)')
ax.set_ylabel('Avg Power Gain (dB)')
ax.grid()


filt_detuning, filt_avg_fit_fidelity = avg_ind_vars(detuning, fit_fidelity)
fig, ax = plt.subplots(figsize = [6,4])
# popt, pcov = curve_fit(line_fit, avg_ind_var(pump_power, phase), avg_ind_var(pump_power, phase), [1, 0])
ax.plot(detuning, np.log10(1-fit_fidelity), '.')
# ax.plot(pump_power, line_fit(pump_power, *popt), '--')
# ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dB}$', (-9,20), fontsize = 20)
ax.set_xlabel('detuning (MHz)')
ax.set_ylabel('$Log_{10}(1-F)$')
ax.grid()


filt_gain, filt_avg_fit_fidelity = avg_ind_vars(avg_power_gain, fit_fidelity)
fig, ax = plt.subplots(figsize = [6,4])
# popt, pcov = curve_fit(line_fit, avg_ind_var(pump_power, phase), avg_ind_var(pump_power, phase), [1, 0])
ax.plot(filt_gain, np.log10(1-np.array(filt_avg_fit_fidelity)), '.')
# ax.plot(pump_power, line_fit(pump_power, *popt), '--')
# ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dB}$', (-9,20), fontsize = 20)
ax.set_xlabel('Amplifier gain (dB)')
ax.set_ylabel('$Log_{10}(1-F)$')
ax.grid()

fig, ax = plt.subplots(figsize = [6,4])
# popt, pcov = curve_fit(line_fit, avg_ind_var(pump_power, phase), avg_ind_var(pump_power, phase), [1, 0])
s = ax.scatter(avg_power_gain, detuning, c = np.log10(1-fit_fidelity), cmap = 'seismic')
# ax.plot(pump_power, line_fit(pump_power, *popt), '--')
# ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dB}$', (-9,20), fontsize = 20)
ax.set_xlabel('Amplifier gain (dB)')
ax.set_ylabel('detuning (MHz)')
ax.grid()
fig.colorbar(s, label = "$Log_{10}(1-F)$")

# fig, ax = plt.subplots(figsize = [6,4])
# popt, pcov = curve_fit(line_fit, pump_power[0:6], avg_power_gain[0:6], [1, 0])
# ax.plot(pump_power, avg_power_gain, '.')
# ax.plot(pump_power, line_fit(pump_power, *popt), '--')
# ax.annotate(f'Slope: {np.round(popt[0], 3)}'+r'$\frac{dB}{dB}$', (-9,20), fontsize = 20)
# ax.grid()
# ax.set_xlabel('Pump Power (dBm)')
# ax.set_ylabel('Power Gain (dB)')