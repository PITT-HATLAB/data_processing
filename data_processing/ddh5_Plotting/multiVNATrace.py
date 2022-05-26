# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:31:32 2022

@author: Hatlab-RRK

purpose: plot all VNA traces in a directory and all of that directory's subdirectories
"""

from plottr.data.datadict_storage import all_datadicts_from_hdf5
from data_processing.Helper_Functions import find_all_ddh5
import matplotlib.pyplot as plt
import numpy as np
#get whatever you want to subtract from the traces you're taking
sb_fp = r'Z:/Data/00_Calibrations/TX_Calibrations/HEMTS/2022-04-18/extra/2022-04-18_0032_C_line_thru_noHEMT/2022-04-18_0032_C_line_thru_noHEMT.ddh5'
# sb_fp = r'Z:/Data/00_Calibrations/TX_Calibrations/Input_Lines_20220418/7_thru_12/cal/2022-04-21_0002_calib_RT_S21/2022-04-21_0002_calib_RT_S21.ddh5'


dd = all_datadicts_from_hdf5(sb_fp)['data']
freqs = dd.extract('power')['frequency']['values']

norm_pows = dd.extract('power')['power']['values']

cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\HEMTS\2022-05-07'
# cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\HEMTS\2022-04-18\VNA'
# cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\Input_Lines_20220418\graph'
# cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\Input_Lines_20220418\7_thru_12\S21'
# cwd = r'Z:\Data\N25_L3_SQ\traces\NVR'
cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\Input_Lines_20220507\S21'
cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\New_Lines_1_2_4_5_20220517'
cal = 0
norm = 0
filepaths = find_all_ddh5(cwd)
fig, ax = plt.subplots(figsize = (8,6))
for filepath in filepaths:
    title_start = filepath.find('base')
    # print(title_start)
    title = filepath[title_start+5: -4]
    print(title)
    dd = all_datadicts_from_hdf5(filepath)['data']
    freqs = dd.extract('power')['frequency']['values']
    if cal: 
        pows = dd.extract('power')['power']['values']-norm_pows
    elif norm: 
        pows = dd.extract('power')['power']['values']
        pows-=pows[0]
        # pows-=pows[0]
    else: 
        pows = dd.extract('power')['power']['values']
    freqfilt = (freqs<20e9)*(freqs>0.5e9)
    ax.plot(freqs[freqfilt]/1e9, pows[freqfilt], label = title)
    
# print(pows[np.isclose(freqs, 6e9, atol = 100e6)])
# ax.set_ylim(-20, 45)1   
ax.set_xlabel('VNA frequency (GHz)')
ax.set_ylabel('VNA Gain (dB)')
ax.set_yticks(np.arange(-100, -20, 5))
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(bbox_to_anchor= (1,1))
ax.grid()

#spectrum analyzer
#%%
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from data_processing.Helper_Functions import find_all_ddh5
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('hatlab')
#get whatever you want to subtract from the traces you're taking
sb_fp = r'Z:/Data/N25_L3_SQ/traces/NVR/2022-04-25/2022-04-25_0004_bp1_NVR_amp_off/2022-04-25_0004_bp1_NVR_amp_off.ddh5'


dd = all_datadicts_from_hdf5(sb_fp)['data']
freqs = dd.extract('power')['frequency']['values']
freqfilt = (freqs<20e9)*(freqs>0e9)
norm_pows = np.average(dd.extract('power')['power']['values'])

cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\HEMTS\2022-05-07'
cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\HEMTS\2022-04-18\VNA'
cwd = r'Z:\Data\N25_L3_SQ\traces\NVR'
cwd = r'Z:\Data\N25_L3_SQ\traces\NVR\2022-04-25\2022-04-25_0003_bp1_NVR'
# cwd = r'Z:\Data\00_Calibrations\TX_Calibrations\HEMTS\2022-04-18\SA'
cal = 1
norm = 0
filepaths = find_all_ddh5(cwd)
fig, ax = plt.subplots()
titles = ["Amplifier on", "Amplifier off"]
for i, filepath in enumerate(filepaths):
    # title_start = filepath[(filepath.find('bp1_NVR')+10):].find('bp1_NVR')
    # print(title_start)
    # title = filepath[title_start: -5]
    # print(title)
    dd = all_datadicts_from_hdf5(filepath)['data']
    freqs = dd.extract('power')['frequency']['values']
    if cal: 
        pows = dd.extract('power')['power']['values']-norm_pows
    else: 
        pows = dd.extract('power')['power']['values']
        
    ax.plot(freqs/1e9, pows, label = titles[i])
    

# ax.set_ylim(-20, 45)1   
ax.set_xlabel('SA frequency (GHz)')
ax.set_ylabel('Noise Visibility (dB)')
ax.set_title("Amplifier Performance")
ax.legend()
# ax.grid()
ax.set_title('NVR')

#noise and Gain on one plot
#%%
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from data_processing.Helper_Functions import find_all_ddh5
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('hatlab')
#get whatever you want to subtract from the noise traces you're taking
sb_fp = r'Z:/Data/N25_L3_SQ/traces/NVR/2022-04-25/2022-04-25_0004_bp1_NVR_amp_off/2022-04-25_0004_bp1_NVR_amp_off.ddh5'
dd = all_datadicts_from_hdf5(sb_fp)['data']
norm_pows = np.average(dd.extract('power')['power']['values'])

gain_filepath = r'Z:/Data/N25_L3_SQ/traces/gain/2022-04-25/2022-04-25_0001_bp1_gain/2022-04-25_0001_bp1_gain.ddh5'
noise_filepath = r'Z:/Data/N25_L3_SQ/traces/NVR/2022-04-25/2022-04-25_0003_bp1_NVR/2022-04-25_0003_bp1_NVR.ddh5'



cal = 1
norm = 0

fig, ax = plt.subplots()

#plot the NVR
dd = all_datadicts_from_hdf5(noise_filepath)['data']
freqs = dd.extract('power')['frequency']['values']

fbounds = [freqs[28], freqs[-1]]
pows = dd.extract('power')['power']['values']-norm_pows
ffiltN = (freqs>fbounds[0])*(freqs<fbounds[1])

ax.plot((freqs[ffiltN]-np.average(freqs[ffiltN]))/1e6, pows[ffiltN], '.', label = "NVR (dB)")

#plot the Gain
dd = all_datadicts_from_hdf5(gain_filepath)['data']
freqs = dd.extract('power')['frequency']['values']

ffilt = (freqs>fbounds[0])*(freqs<fbounds[1])

pows = dd.extract('power')['power']['values']
ax.plot((freqs[ffilt]-np.average(freqs[ffilt]))/1e6, pows[ffilt], label = 'Gain (dB)')


# ax.set_ylim(-20, 45)1   
ax.set_xlabel('Frequency (MHz)')
# ax.set_ylabel('Noise Visibility (dB)')
ax.set_title("Amplifier Performance")
ax.legend()
# ax.grid()