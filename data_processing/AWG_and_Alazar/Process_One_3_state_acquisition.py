# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:18:59 2021

@author: Ryan Kaufman - Hatlab
"""
from plottr.apps.autoplot import main
from plottr.data import datadict_storage as dds, datadict as dd
from data_processing.signal_processing import Pulse_Processing_utils as PU
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  
import warnings
warnings.filterwarnings("ignore")
import os
def find_all_ddh5(cwd): 
    dirs = os.listdir(cwd)
    filepaths = []
    for path in dirs: 
        try:
            subs = os.listdir(cwd+'\\'+path)
            for sub in subs: 
                print(sub)
                if sub.split('.')[-1] == 'ddh5':  
                    filepaths.append(cwd+'\\'+path+'\\'+sub)
                else: 
                    for subsub in os.listdir(cwd+'\\'+path+'\\'+sub):
                        if subsub.split('.')[-1] == 'ddh5':  
                            filepaths.append(cwd+'\\'+path+'\\'+sub+'\\'+subsub)
        except: #usually because the files are one directory higher than you'd expect
            if path.split('.')[-1] == 'ddh5':  
                    filepaths.append(cwd+'\\'+path)
    return filepaths

#%%sample one file to check things

filepath = r'G:/My Drive/shared/Amplifier_Response_Data/2021-09-30_0030_3_state_deep_sat_40dB_att_8us_time_Rep_4__.ddh5'

# PU.get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False, hist_scale = 0.01, records_per_pulsetype = 3870*2)
# IQ_offset = PU.get_IQ_offset_from_filepath(filepath, plot = False, hist_scale = 0.002, records_per_pulsetype = 3840*2)
# PU.get_fidelity_from_filepath_3_state(filepath, plot = True, hist_scale = 0.05, records_per_pulsetype = 2562, state_relabel = 0, bin_start = 50, bin_stop = 400)
PU.get_fidelity_from_filepath_3_state(filepath, plot = True, hist_scale = 0.7, records_per_pulsetype = 7686//3, state_relabel = 0, bin_start = 50, bin_stop = 400, fit = False)
IQ_offset = (0,0)

#%%
# original bias point (closest to bp2)

filepath = r'Z:\Data\Hakan\SA_3C1_3221_7GHz\signal_power_sweep\2021-11-18\2021-11-18_0033_amp_gain_test_Sig_Volt_0.05_V_Rep_2__\2021-11-18_0033_amp_gain_test_Sig_Volt_0.05_V_Rep_2__.ddh5'
filepath = r'Z:\Data\Hakan\SA_3C1_3221_7GHz\signal_power_sweep\2021-11-18\2021-11-18_0036_amp_gain_test_Sig_Volt_0.1_V_Rep_0__\2021-11-18_0036_amp_gain_test_Sig_Volt_0.1_V_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SA_3C1_3221_7GHz\signal_power_sweep\2021-11-18\2021-11-18_0075_amp_gain_test_Sig_Volt_0.45_V_Rep_4__\2021-11-18_0075_amp_gain_test_Sig_Volt_0.45_V_Rep_4__.ddh5'

fidelity = PU.extract_3pulse_histogram_from_filepath(filepath, 
                                            numRecords =  7686, 
                                            IQ_offset = (0,0), 
                                            plot = True, 
                                            hist_scale = 0.04, 
                                            fit = True,
                                            boxcar = False,
                                            bc_window = [50, 150],
                                            lpf = False, 
                                            lpf_wc = 15e6, 
                                            record_track = True, 
                                            tuneup_plots = True)
print(fidelity)
#%% extract the average phase difference between records from filepaths
#extract 3 pulse noise for the 
filepaths = find_all_ddh5(r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\detuning_sweep_fine\2021-10-14')
phases = []
for filepath in filepaths:
    phases.append(PU.extract_3pulse_phase_differences_from_filepath(filepath,
                                                numRecords =  7686,
                                                window = [50, 205], 
                                                scale = 2))
print("Average Powers: ", phases)
#%%plot the phases wrt detuning
SG_pow = 7.78 #without 10dB att and with switch + cable
x = np.arange(-0.5, 0.5, 0.05)+0.1
xlog = 20*np.log10(x)
x_name = 'Signal Detuning from pump/2 (MHz)'
fig, ax = plt.subplots(1, 1)

ax.set_title(r"average phase difference between records")
yplt = np.array(phases)
ax.plot(x, yplt, '.')
# ax.plot(x, x, '.', label = label)
ax.grid()
# ax.set_aspect(1)
ax.set_xlabel(x_name)
ax.set_ylabel('Phase difference (degrees)')
ax.legend()
# if i == 1: 

#%%
#extract 3 pulse noise for the 
filepaths = find_all_ddh5(r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_sweep_no_sig__extra_fine_high_power')
pwrs = []
for filepath in filepaths:
    pwrs.append(PU.extract_3pulse_noise_from_filepath(filepath,
                                                numRecords =  7686,
                                                window = [50, 205]))
print("Average Powers: ", pwrs)
# PU.extract_noise_power()
#%%noise plotting
pwr = pwrs
labels = ["G", "E", "F"]
SG_pow = 7.78 #without 10dB att and with switch + cable
x = np.arange(SG_pow-1, SG_pow+1, 0.05)
xlog = 20*np.log10(x)
x_name = 'Pump input power (dBm RT)'
fig, ax = plt.subplots(1, 1)

ax.set_title(r"$\sigma_V$")
yplt = np.array(pwr)
ax.plot(x, yplt, '.')
# ax.plot(x, x, '.', label = label)
ax.grid()
# ax.set_aspect(1)
ax.set_xlabel(x_name)
ax.set_ylabel('Noise std dev (mV)')
ax.legend()
# if i == 1: 
    # print("peak G input gain power: ", x[np.argmin(np.abs(np.max(yplt)-yplt))])

#%%
#wait_time_sweep
# filepaths = find_all_ddh5(r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\wait_time_sweep_fine')
#input_power_sweep
filepaths = find_all_ddh5(r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\signal_power_sweep')
pwrs = []
for filepath in filepaths:
    pwrs.append(PU.extract_3pulse_pwr_from_filepath(filepath,
                                                numRecords =  7686,
                                                window = [50, -1]))
print("Average Powers: ", pwrs)
#%%gain plotting
labels = ["G", "E", "F"]
x = np.arange(0.05, 1.5, 0.05)
xlog = 20*np.log10(x)
x_name = 'Signal input power (dBV)'
fig, ax = plt.subplots(1, 1)
for i, pwr in enumerate(np.array(pwrs).T): 
    # pwrs = np.array(pwrs)
    label = labels[i]
    ax.set_title(r"$S_{21} = 20log(V_{out}/V_{in})$")
    yplt = 20*np.log10(np.array(pwr)/x)
    ax.plot(xlog, yplt, '.', label = label)
    # ax.plot(x, x, '.', label = label)
    ax.grid()
    # ax.set_aspect(1)
    ax.set_xlabel(x_name)
    ax.set_ylabel('S21 (dB)')
    ax.legend()
    if i == 1: 
        print("peak G input gain power: ", x[np.argmin(np.abs(np.max(yplt)-yplt))])


#%%exponential decay fitting 
from scipy.optimize import curve_fit
labels = ["G", "E", "F"]
for i, pwr in enumerate(np.array(pwrs.T)): 
    # pwrs = np.array(pwrs)
    label = labels[i]
    fig, ax = plt.subplots(1, 1)
    
    ax.set_title(f"{label} Fit")
    ax.plot(np.arange(0, 500, 10), np.array(pwr)*1000, '.')
    ax.grid()
    ax.set_xlabel('Wait time (us)')
    ax.set_ylabel('Average magnitude (mV)')
    x = np.arange(0, 500, 10)
    fit_func = lambda t, A, B: A*np.exp(B*t)
    popt, pcov = curve_fit(fit_func, x, pwr, p0 = (75, -0.005))
    ax.plot(x, fit_func(x, *popt)*1000, '--', color = 'orange')
    ax.annotate(f"Decay time: {np.round(-1/popt[1])} us", (100, 54))
    plt.show()
