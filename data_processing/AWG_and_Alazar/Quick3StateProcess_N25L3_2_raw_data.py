# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:11:55 2022

@author: Hatlab-RRK

Purpose: create a neat, almost-executable file that can quickly plot a 3-state pulse file, and have the option to do just histograms, 
or additionally try to fit using majority vote and give classification accuracy
"""
import data_processing.AWG_and_Alazar.Pulse_Processing_utils_raw_data as pulseUtils
import os
import matplotlib.pyplot as plt
plt.style.use('hatlab')
import numpy as np

# datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\fixed\signal_power_sweep_-30MHz_detuned\2022-05-25_0001_pwr_swp_0dB_att_Amp_0__.ddh5'
#amp on
datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/2022-05-26_0007_pwr_swp_0dB_att_Amp_1__.ddh5'

# datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\init_amp_test_replug_clock\2022-05-25_0003_pwr_swp_0dB_att_pump_off_Sig_Volt_1.244_V_.ddh5'
nr = 7686

time, signal_arr, ref_arr = pulseUtils.Process_One_Acquisition_3_state(datapath)
#%%custom stats
for trace in [0,1,2]: 
    Pvar, Pvar_fit, Pavg, Ivar, Qvar = plot_custom_stats_from_filepath(datapath, debug = True, trace = trace, fit = 1, timeslice = 100)
    print("Variance (numpy):", Pvar, "\nVariance (fit): ", Pvar_fit, "\nAverage: ", Pavg)

#%%old stats
plot_stats_from_filepath(datapath, plt_avg = 0, plt_var = 1, vscale = 100, plot = 1)

#%%

file_arr_amp_off = [
    "Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0014_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0001_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0002_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0003_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0004_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0005_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0006_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0007_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0008_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0009_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0010_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0011_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0012_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_higher_pwr/2022-05-05_0013_pwr_sweep_20dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.25_V_.ddh5"
    
    ]

file_arr_amp_on = [
    "Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0014_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0001_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0002_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0003_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0004_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0005_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0006_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0007_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0008_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0009_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0010_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0011_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0012_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_20dB_att/2022-05-05_0013_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.25_V_.ddh5"
    ]

Pvar_arr, Pvar_fit_arr, Pavg_arr, Ivar_arr, Qvar_arr = [],[],[],[],[]
for fp in file_arr_amp_off: 
    Pvar, Pvar_fit, Pavg, Ivar, Qvar = plot_custom_stats_from_filepath(fp, debug = True, trace = 0)
    # print("Variance (numpy):", Pvar, "\nVariance (fit): ", Pvar_fit, "\nAverage: ", Pavg)

    Pvar_arr.append(Pvar)
    Pvar_fit_arr.append(Pvar_fit)
    Pavg_arr.append(Pavg)
    Ivar_arr.append(Ivar)
    Qvar_arr.append(Qvar)


#%%
plt.plot(Pavg_arr, Ivar_arr, '.')
plt.plot(Pavg_arr, Qvar_arr, '.')
# plt.plot(Pavg_arr, Pvar_arr, '.', label = 'numpy variance')
# plt.plot(Pavg_arr, Pvar_fit_arr, '.', label = 'fit_variance')
# plt.xlim(0, 0.04)
# plt.ylim(0, 0.0031)
plt.legend()

 #%%
pwr_file_arr = [
    "Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0022_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0009_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0010_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0011_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0012_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0013_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0014_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0015_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0016_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0017_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0018_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0019_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0020_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0021_pwr_sweep_0dB_att_pump_pwr_9.76_dBm_LO_freq_5914000000.0_Hz_Sig_Volt_1.25_V_.ddh5"
    ]
pwr_file_arr = [
    "Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0051_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0052_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0053_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0054_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0055_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0056_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0057_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0058_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0059_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0060_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0061_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0062_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0063_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_saturation_0.4mA_sweep_0dB_att/2022-05-05_0064_pwr_sweep_0dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.35_V_.ddh5"
    ]
pwr_file_arr = [
    "Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0028_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0015_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0016_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0017_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0018_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0019_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0020_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0021_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0022_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0023_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0024_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0025_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0026_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0027_pwr_sweep_10dB_att_pump_pwr_9.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.25_V_.ddh5"
    ]

pwr_file_arr = [
    "Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0056_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0043_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0044_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0045_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.25_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0046_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.35_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0047_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.45_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0048_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.55_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0049_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.65_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0050_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.75_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0051_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.85_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0052_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_0.95_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0053_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.05_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0054_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.15_V_.ddh5",
"Z:/Data/N25_L3_SQ/BP1/amp_was_actually_off/amp_saturation_0.4mA_sweep_even_higher_pwr/2022-05-05_0055_pwr_sweep_10dB_att_pump_pwr_10.76_dBm_LO_freq_5917000000.0_Hz_Sig_Volt_1.25_V_.ddh5"
    ]
Gvar_arr, Evar_arr, Fvar_arr = [],[],[]
Gavg_arr, Eavg_arr, Favg_arr = [],[],[]
for i, fp in enumerate(pwr_file_arr): 
    print(i, fp)
    [Gvar,Gavg], [Evar,Eavg], [Fvar,Favg] = plot_stats_from_filepath(fp, vscale = 1, plot = 0)
    Gvar_arr.append(Gvar)
    Evar_arr.append(Evar)
    Fvar_arr.append(Fvar)
    
    Gavg_arr.append(Gavg)
    Eavg_arr.append(Eavg)
    Favg_arr.append(Favg)
    
#%%
plt.plot(np.array(Gavg_arr),np.array(Gvar_arr), '.')