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

IQ_offset =  np.array((0,0))
# records_per_pulsetype = 3870
cf = 6171427180.18
# amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/Pump_power_sweeps/1/2021-06-30/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094.ddh5'
# amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/signal_power_sweeps/1_initial_guess/2021-07-06/2021-07-06_0003_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.0_V_Phase_0.0_rad_/2021-07-06_0003_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.0_V_Phase_0.0_rad_.ddh5'

# filepath = r'G:/My Drive/shared/Amplifier_Response_Data/Data/Pump_pwr_detuning_sweeps/2021-07-07/2021-07-07_0409_Amp_1__pwr_-8.83_dBm_LO_freq_6172127180.18_Hz_Phase_0.0_rad_/2021-07-07_0409_Amp_1__pwr_-8.83_dBm_LO_freq_6172127180.18_Hz_Phase_0.0_rad_.ddh5'

# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0002_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_1__/2021-09-14_0002_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_1__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0027_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_1__/2021-09-14_0027_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_1__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0034_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_3__/2021-09-14_0034_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_3__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0038_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_2__/2021-09-14_0038_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_2__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0044_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_3__/2021-09-14_0044_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_3__.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0046_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_0__/2021-09-14_0046_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_0__.ddh5'

# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0001_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_0__/2021-09-14_0001_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0028_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_2__/2021-09-14_0028_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_2__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0031_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_0__/2021-09-14_0031_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0036_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_0__/2021-09-14_0036_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0042_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_1__/2021-09-14_0042_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_1__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0047_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_1__/2021-09-14_0047_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_1__.ddh5'

# PU.get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False, hist_scale = 0.01, records_per_pulsetype = 3870*2)
# IQ_offset = PU.get_IQ_offset_from_filepath(filepath, plot = False, hist_scale = 0.002, records_per_pulsetype = 3840*2)
data_fidelity, fit_fidelity, even_fit, odd_fit = PU.get_fidelity_from_filepath_3_state(filepath, plot = True, hist_scale = 0.02, records_per_pulsetype = 2562, state_relabel = 0)
print(data_fidelity, fit_fidelity)
IQ_offset = (0,0)
