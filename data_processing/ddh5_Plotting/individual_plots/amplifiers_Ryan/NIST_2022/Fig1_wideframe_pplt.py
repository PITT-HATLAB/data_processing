# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:31:32 2022

@author: Hatlab-RRK

purpose: plot all VNA traces in a directory and all of that directory's subdirectories
"""
#noise and Gain on one plot
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from data_processing.Helper_Functions import find_all_ddh5
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('hatlab')
# Rebuild the matplotlib font cache
#get whatever you want to subtract from the noise traces you're taking
#old one: 
# sb_fp = r'Z:/Data/N25_L3_SQ/traces/NVR/2022-05-02_0008_NVR_amp_off.ddh5'
#new one: 
# sb_fp = r'Z:/Data/N25_L3_SP_2/traces/20dB/2022-05-23_0003_NVR_amp_off.ddh5'
# dd = all_datadicts_from_hdf5(sb_fp)['data']
# norm_pows = np.average(dd.extract('power')['power']['values'])

# gain_filepath = r'Z:/Data/N25_L3_SQ/traces/gain/2022-04-25/2022-04-25_0001_bp1_gain/2022-04-25_0001_bp1_gain.ddh5'
#old 6MHz wide:
# gain_filepath = r'Z:/Data/N25_L3_SQ/traces/gain/2022-05-02/2022-05-02_0002_gain_2/2022-05-02_0002_gain_2.ddh5'
# noise_filepath = r'Z:\Data\N25_L3_SQ\traces\NVR\2022-05-02_0009_NVR_amp_on.ddh5'

#new 26MHz wide: 
# gain_filepath = r'Z:/Data/N25_L3_SP_2/traces/20dB/2022-05-23_0001_gain_0.16mA.ddh5'
# noise_filepath = r'Z:/Data/N25_L3_SP_2/traces/20dB/2022-05-23_0002_NVR_amp_on.ddh5' 

#for a bunch of gain curves
gain_fps = [r'Z:/Data/N25_L3_SP_2/traces/18dB_wide/2022-05-24_0001_0.16mA.ddh5', 
            r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0001_0.16mA.ddh5', 
            r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0002_0.16mA.ddh5']

sat_fps = [r'Z:/Data/N25_L3_SP_2/traces/18dB_wide/2022-05-24_0002_sat_amp_on.ddh5', 
           r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0002_sat_amp_on.ddh5', 
           r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0005_sat_amp_on.ddh5']

sat_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/18dB_wide/2022-05-24_0003_sat_amp_off.ddh5', 
              r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0003_sat_amp_off.ddh5', 
              r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0006_sat_amp_off.ddh5']

NVR_fps = [r'Z:/Data/N25_L3_SP_2/traces/18dB_wide/2022-05-24_0005_NVR_amp_on.ddh5', 
           r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0005_NVR_amp_on.ddh5', 
           r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0003_NVR_amp_on.ddh5']

NVR_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/18dB_wide/2022-05-24_0006_NVR_amp_off.ddh5', 
              r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0004_NVR_amp_off.ddh5', 
              r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0004_NVR_amp_off.ddh5']
# ncols = 3
# bbox = (1.1, 1.2)
# fbounds_ind = [0, -1]

# #for only the most clean
# gain_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0002_0.16mA.ddh5']
# sat_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0005_sat_amp_on.ddh5']
# sat_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0006_sat_amp_off.ddh5']
# NVR_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0003_NVR_amp_on.ddh5']
# NVR_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_narrow/2022-05-24_0004_NVR_amp_off.ddh5']
# fbounds_ind = [800, 1200]
# #wide 20dB
# gain_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0001_0.16mA.ddh5']
# sat_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0002_sat_amp_on.ddh5']
# sat_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0003_sat_amp_off.ddh5']
# NVR_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0005_NVR_amp_on.ddh5']
# NVR_sb_fps = [r'Z:/Data/N25_L3_SP_2/traces/20dB_wide_2/2022-05-24_0004_NVR_amp_off.ddh5']

ncols = 2
bbox = (0.75, 1.1)
fbounds_ind = [100, 1900]


gainfig, gain_ax = plt.subplots()
satfig, sat_ax = plt.subplots()
skip = 20
satskip = 100
gain_colors = ['blue', 'green', 'red']
noise_colors = ['darkblue', 'darkgreen', 'darkred']

satfig, sat_ax = plt.subplots()
for i, (gain_fp, sat_fp, sat_sb_fp, NVR_fp, NVR_sb_fp) in enumerate(zip(gain_fps, sat_fps, sat_sb_fps, NVR_fps, NVR_sb_fps)):
    gain_fig, gain_ax = plt.subplots()
    nvr_fig, nvr_ax = plt.subplots()
    gain_ax.set_ylim(10, 23)
    nvr_ax.set_ylim(0, 10)
    gain_colors = ['blue', 'green', 'red']
    noise_colors = ['darkblue', 'darkgreen', 'darkred']
    print("\n ", i)
    
    #plot the Gain
    dd = all_datadicts_from_hdf5(gain_fp)['data']
    freqs = dd.extract('power')['frequency']['values']
    
    fbounds = [freqs[fbounds_ind[0]], freqs[fbounds_ind[1]]]
    
    ffilt = (freqs>fbounds[0])*(freqs<fbounds[1])
    pows = dd.extract('power')['power']['values']
    
    gain_ax.plot(((freqs[ffilt]-np.average(freqs[ffilt]))/1e6)[::skip], pows[ffilt][::skip], 's', label = 'Gain (dB)', color = gain_colors[i])
    gain_ax.plot(((freqs[ffilt]-np.average(freqs[ffilt]))/1e6)[::skip], pows[ffilt][::skip], 's-', color = gain_colors[i])
    
    #plot the NVR
    sb_fp = NVR_sb_fp
    fp = NVR_fp
    
    dd = all_datadicts_from_hdf5(sb_fp)['data']
    norm_pows = dd.extract('power')['power']['values']
    
    dd = all_datadicts_from_hdf5(fp)['data']
    Nfreqs = dd.extract('power')['frequency']['values']
    

    Npows = dd.extract('power')['power']['values']-norm_pows
    ffiltN = (Nfreqs>fbounds[0])*(Nfreqs<fbounds[1])
    nvr_ax.plot((Nfreqs[ffiltN]-np.average(Nfreqs[ffiltN]))[::skip]/1e6, Npows[ffiltN][::skip], 'o', label = "NVR (dB)", color = noise_colors[i])
    nvr_ax.plot((Nfreqs[ffiltN]-np.average(Nfreqs[ffiltN]))[::skip]/1e6, Npows[ffiltN][::skip], 'o-', color = noise_colors[i])
    

    

    # gain_ax.set_title('Gain (dB)')
    # nvr_ax.set_title('NVR (dB)')
    gain_ax.set_xticks([-20, 20])
    nvr_ax.set_xticks([-20,20])
    nvr_ax.set_yticks([0,2,4,6,8])
    gain_ax.set_yticks([10,12,14,16,18,20,22])
    if i == 2: 
        gain_ax.set_ylabel('Gain (dB)')
        nvr_ax.set_ylabel('NVR (dB)')
        gain_ax.set_yticklabels(['', 12, '',16,'',20,''])
        nvr_ax.set_yticklabels(['',2,'','',8])
    else: 
        gain_ax.set_yticklabels(['', '', '','','','',''])
        nvr_ax.set_yticklabels(['','','','',''])
    if i == 1: 
        nvr_ax.set_xlabel('Frequency Detuning (MHz)')

    # ax.set_title("Amplifier Performance")
    # gain_ax.legend(handletextpad = -0.5, bbox_to_anchor = bbox, ncol = ncols, columnspacing = 0.1)
    
    #saturation
    #old: 
    # filepath = r'Z:/Data/N25_L3_SQ/traces/sat/2022-05-02/2022-05-02_0004_sat_2/2022-05-02_0004_sat_2.ddh5'
    
    
        
        
    sb_fp = sat_sb_fp
    dd = all_datadicts_from_hdf5(sb_fp)['data']
    norm_pows = np.average(dd.extract('power')['power']['values'])
    
    dd = all_datadicts_from_hdf5(sat_fp)['data']
    pows = dd.extract('power')['power']['values']
    freqs = dd.extract('power')['frequency']['values']
    
    #plot the LO leakage vs power
    att = 85
    sat_ax.plot((freqs-att)[::satskip], (pows-norm_pows)[::satskip], 's-', color = gain_colors[i])
    sat_ax.set_xlabel('Signal power (dBm)')
    # ax.set_ylabel('Gain (dB)')
    sat_ax.legend()
    sat_ax.hlines([19, 17], np.min(freqs)-att, np.max(freqs)-att, linestyle = '--', color = 'k')
    sat_ax.set_xticks([-120, -110, -100, -90, -80])
    sat_ax.set_xticklabels([-120, '', '', -90, ''])
    # ax.vlines([-90.5], np.min(pows-norm_pows), np.max(pows-norm_pows), linestyle = '--', color = 'k')
    # ax.grid()
    # ax.set_title('0.06mA, 8.65dBm RT, +277kHZ generator detuning: 6.6MHz BW')


