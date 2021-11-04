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
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0027_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_1__/2021-09-14_0027_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_1__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0034_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_3__/2021-09-14_0034_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_3__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0038_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_2__/2021-09-14_0038_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_2__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0044_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_3__/2021-09-14_0044_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_3__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep/2021-09-14/2021-09-14_0046_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_0__/2021-09-14_0046_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_0__.ddh5'

# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0001_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_0__/2021-09-14_0001_3_state_40dB_att_Amp_0__pwr_-7.0_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0028_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_2__/2021-09-14_0028_3_state_40dB_att_Amp_1__pwr_-7.0_dBm_Rep_2__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0031_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_0__/2021-09-14_0031_3_state_40dB_att_Amp_1__pwr_-6.75_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0036_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_0__/2021-09-14_0036_3_state_40dB_att_Amp_1__pwr_-6.5_dBm_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0042_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_1__/2021-09-14_0042_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_1__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/gen_pwr_sweep_+500kHz/2021-09-14/2021-09-14_0047_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_1__/2021-09-14_0047_3_state_40dB_att_Amp_1__pwr_-6.0_dBm_Rep_1__.ddh5'

#loopback for sat_discriminator
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\loopbacks\2021-09-30\2021-09-30_0009_3_state_loopback_0dB_att_Rep_0__\2021-09-30_0009_3_state_loopback_0dB_att_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0002_3_state_deep_sat_40dB_att_Rep_0__\2021-09-30_0002_3_state_deep_sat_40dB_att_Rep_0__.ddh5'

#in order of increasing power
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0003_3_state_deep_sat_40dB_att_Sig_Volt_0.2_V_Rep_0__\2021-09-30_0003_3_state_deep_sat_40dB_att_Sig_Volt_0.2_V_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0004_3_state_deep_sat_40dB_att_Sig_Volt_0.25_V_Rep_0__\2021-09-30_0004_3_state_deep_sat_40dB_att_Sig_Volt_0.25_V_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0005_3_state_deep_sat_40dB_att_Sig_Volt_0.3_V_Rep_0__\2021-09-30_0005_3_state_deep_sat_40dB_att_Sig_Volt_0.3_V_Rep_0__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0011_3_state_deep_sat_40dB_att_Sig_Volt_0.6_V_Rep_0__\2021-09-30_0011_3_state_deep_sat_40dB_att_Sig_Volt_0.6_V_Rep_0__.ddh5'

#longer time: 
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0017_3_state_deep_sat_40dB_att_8us_time_Rep_0__\2021-09-30_0017_3_state_deep_sat_40dB_att_8us_time_Rep_0__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-09-30\2021-09-30_0030_3_state_deep_sat_40dB_att_8us_time_Rep_4__\2021-09-30_0030_3_state_deep_sat_40dB_att_8us_time_Rep_4__.ddh5'

#SWEEPING power
#0.55V
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-01\2021-10-01_0076_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.55_V_\2021-10-01_0076_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.55_V_.ddh5'

#0.6V
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-01\2021-10-01_0077_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.6_V_\2021-10-01_0077_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.6_V_.ddh5'

#0.95V
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-01\2021-10-01_0084_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.95_V_\2021-10-01_0084_3_state_deep_sat_40dB_att_2V_Sig_Volt_0.95_V_.ddh5'

#1.1V
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-01\2021-10-01_0088_3_state_deep_sat_40dB_att_2V_Sig_Volt_1.15_V_\2021-10-01_0088_3_state_deep_sat_40dB_att_2V_Sig_Volt_1.15_V_.ddh5'

#WTF Trigger?

import easygui
# filepath = easygui.fileopenbox(default = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-01\*')
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-04\2021-10-04_0005_3_state_deep_sat_40dB_att_2V_Rep_4__\2021-10-04_0005_3_state_deep_sat_40dB_att_2V_Rep_4__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-04\2021-10-04_0004_3_state_deep_sat_40dB_att_2V_Rep_3__\2021-10-04_0004_3_state_deep_sat_40dB_att_2V_Rep_3__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-04\2021-10-04_0003_3_state_deep_sat_40dB_att_2V_Rep_2__\2021-10-04_0003_3_state_deep_sat_40dB_att_2V_Rep_2__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-04\2021-10-04_0002_3_state_deep_sat_40dB_att_2V_Rep_1__\2021-10-04_0002_3_state_deep_sat_40dB_att_2V_Rep_1__.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\saturation_discriminator\2021-10-04\2021-10-04_0001_3_state_deep_sat_40dB_att_2V_Rep_0__\2021-10-04_0001_3_state_deep_sat_40dB_att_2V_Rep_0__.ddh5'

filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\digitizer_troubleshooting\2021-10-06\2021-10-06_0028_loopback_10dB_att_Rep_0__\2021-10-06_0028_loopback_10dB_att_Rep_0__.ddh5'

filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\deep_saturation_phase_sweep\2021-10-06\2021-10-06_0001_40dB_att_Phase_0.0_rad_\2021-10-06_0001_40dB_att_Phase_0.0_rad_.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_6.064GHz\3_state\deep_saturation_phase_sweep\amp_on\2021-10-06\2021-10-06_0013_40dB_att_Phase_0.0_rad_\2021-10-06_0013_40dB_att_Phase_0.0_rad_.ddh5'

#phase sweep
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0054_40dB_att_6us_Phase_0.0_rad_/2021-10-06_0054_40dB_att_6us_Phase_0.0_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0055_40dB_att_6us_Phase_0.524_rad_/2021-10-06_0055_40dB_att_6us_Phase_0.524_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0056_40dB_att_6us_Phase_1.047_rad_/2021-10-06_0056_40dB_att_6us_Phase_1.047_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0057_40dB_att_6us_Phase_1.571_rad_/2021-10-06_0057_40dB_att_6us_Phase_1.571_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0058_40dB_att_6us_Phase_2.094_rad_/2021-10-06_0058_40dB_att_6us_Phase_2.094_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0059_40dB_att_6us_Phase_2.618_rad_/2021-10-06_0059_40dB_att_6us_Phase_2.618_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0060_40dB_att_6us_Phase_3.142_rad_/2021-10-06_0060_40dB_att_6us_Phase_3.142_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0061_40dB_att_6us_Phase_3.665_rad_/2021-10-06_0061_40dB_att_6us_Phase_3.665_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0062_40dB_att_6us_Phase_4.189_rad_/2021-10-06_0062_40dB_att_6us_Phase_4.189_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0063_40dB_att_6us_Phase_4.712_rad_/2021-10-06_0063_40dB_att_6us_Phase_4.712_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0064_40dB_att_6us_Phase_5.236_rad_/2021-10-06_0064_40dB_att_6us_Phase_5.236_rad_.ddh5'
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/3_state/deep_saturation_phase_sweep/amp_on/2021-10-06/2021-10-06_0065_40dB_att_6us_Phase_5.76_rad_/2021-10-06_0065_40dB_att_6us_Phase_5.76_rad_.ddh5'

filepath = r'G:/My Drive/shared/Amplifier_Response_Data/2021-09-30_0030_3_state_deep_sat_40dB_att_8us_time_Rep_4__.ddh5'

# PU.get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False, hist_scale = 0.01, records_per_pulsetype = 3870*2)
# IQ_offset = PU.get_IQ_offset_from_filepath(filepath, plot = False, hist_scale = 0.002, records_per_pulsetype = 3840*2)
# PU.get_fidelity_from_filepath_3_state(filepath, plot = True, hist_scale = 0.05, records_per_pulsetype = 2562, state_relabel = 0, bin_start = 50, bin_stop = 400)
PU.get_fidelity_from_filepath_3_state(filepath, plot = True, hist_scale = 0.7, records_per_pulsetype = 7686//3, state_relabel = 0, bin_start = 50, bin_stop = 400, fit = False)
IQ_offset = (0,0)

#%%
# original bias point (closest to bp2)
# filepath = r'G:/My Drive/shared/Amplifier_Response_Data/2021-09-30_0030_3_state_deep_sat_40dB_att_8us_time_Rep_4__.ddh5'
# filepath = r'G:/My Drive/shared/Amplifier_Response_Data/Data/20210927_3state/amplifier_on/2021-09-14_0044_3_state_40dB_att_Amp_1__pwr_-6.25_dBm_Rep_3__.ddh5'

#bp3
# loopback
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\loopback\2021-10-13\2021-10-13_0005_bp3_loopback_Rep_0__\2021-10-13_0005_bp3_loopback_Rep_0__.ddh5'
#oscillating states - amp off
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\loopback\2021-10-13\2021-10-13_0006_bp3_40dbAtt_4us_1.5V_Rep_0__\2021-10-13_0006_bp3_40dbAtt_4us_1.5V_Rep_0__.ddh5'
#oscillating states - amp on
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\loopback\2021-10-13\2021-10-13_0007_bp3_40dbAtt_4us_1.5V_Rep_0__\2021-10-13_0007_bp3_40dbAtt_4us_1.5V_Rep_0__.ddh5'
#more conservative rearming time
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\wait_time_sweep\2021-10-13\2021-10-13_0001_bp3_40dbAtt_4us_1.5V_Trigger_wait_0_us_\2021-10-13_0001_bp3_40dbAtt_4us_1.5V_Trigger_wait_0_us_.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\wait_time_sweep\2021-10-13\2021-10-13_0002_bp3_40dbAtt_4us_1.5V_Trigger_wait_100_us_\2021-10-13_0002_bp3_40dbAtt_4us_1.5V_Trigger_wait_100_us_.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\wait_time_sweep\2021-10-13\2021-10-13_0003_bp3_40dbAtt_4us_1.5V_Trigger_wait_200_us_\2021-10-13_0003_bp3_40dbAtt_4us_1.5V_Trigger_wait_200_us_.ddh5'
filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\wait_time_sweep\2021-10-13\2021-10-13_0005_bp3_40dbAtt_4us_1.5V_Trigger_wait_400_us_\2021-10-13_0005_bp3_40dbAtt_4us_1.5V_Trigger_wait_400_us_.ddh5'

#picking and choosing from the sweep
#0.7V

# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_sweep_no_sig_fine\2021-10-13\2021-10-13_0011_bp3_pump_sweep_no_sig_pump_pwr_7.78_dBm_\2021-10-13_0011_bp3_pump_sweep_no_sig_pump_pwr_7.78_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0010_bp3_pump_sweep_no_sig_pump_pwr_7.28_dBm_/2021-10-13_0010_bp3_pump_sweep_no_sig_pump_pwr_7.28_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0009_bp3_pump_sweep_no_sig_pump_pwr_6.78_dBm_/2021-10-13_0009_bp3_pump_sweep_no_sig_pump_pwr_6.78_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0008_bp3_pump_sweep_no_sig_pump_pwr_6.28_dBm_/2021-10-13_0008_bp3_pump_sweep_no_sig_pump_pwr_6.28_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0006_bp3_pump_sweep_no_sig_pump_pwr_5.28_dBm_/2021-10-13_0006_bp3_pump_sweep_no_sig_pump_pwr_5.28_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0001_bp3_pump_sweep_no_sig_pump_pwr_2.78_dBm_/2021-10-13_0001_bp3_pump_sweep_no_sig_pump_pwr_2.78_dBm_.ddh5'

# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_no_Rb_clock\2021-10-13\2021-10-13_0001_bp3_pump_sweep_no_sig_Rep_0__\2021-10-13_0001_bp3_pump_sweep_no_sig_Rep_0__.ddh5'

# #low power input
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/signal_power_sweep/2021-10-13/2021-10-13_0004_bp3_40dbAtt_4us_1.5V_Sig_Volt_0.2_V_/2021-10-13_0004_bp3_40dbAtt_4us_1.5V_Sig_Volt_0.2_V_.ddh5'

# #no signal, at crit pump power
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig_fine/2021-10-13/2021-10-13_0011_bp3_pump_sweep_no_sig_pump_pwr_7.78_dBm_/2021-10-13_0011_bp3_pump_sweep_no_sig_pump_pwr_7.78_dBm_.ddh5'

# #signal, but compensating 200Hz drift with offset in reference generator
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_no_Rb_clock\2021-10-14\2021-10-14_0001_bp3_SC9_detuning_Rep_0__\2021-10-14_0001_bp3_SC9_detuning_Rep_0__.ddh5' 
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_no_Rb_clock\2021-10-14\2021-10-14_0002_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0002_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_no_Rb_clock\2021-10-14\2021-10-14_0003_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0003_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_demod_detuning\2021-10-14\2021-10-14_0001_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0001_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_demod_detuning\2021-10-14\2021-10-14_0002_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0002_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_demod_detuning\2021-10-14\2021-10-14_0003_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0003_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_demod_detuning\2021-10-14\2021-10-14_0004_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0004_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:\Data\Hakan\SH_5B1_SS_Gain_bp3\3_state\bp3\pulsed_pump_demod_detuning\2021-10-14\2021-10-14_0005_bp3_SC9_detuning_0Hz_Rep_0__\2021-10-14_0005_bp3_SC9_detuning_0Hz_Rep_0__.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/detuning_sweep_fine/2021-10-14/2021-10-14_0009_bp3_LO_detuning_0.3V_LO_freq_6800000000.0_Hz_/2021-10-14_0009_bp3_LO_detuning_0.3V_LO_freq_6800000000.0_Hz_.ddh5'

# #looking at just pump pulsing again
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig__extra_fine_high_power/2021-10-13/2021-10-13_0016_bp3_pump_sweep_no_sig_pump_pwr_7.53_dBm_/2021-10-13_0016_bp3_pump_sweep_no_sig_pump_pwr_7.53_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig__extra_fine_high_power/2021-10-13/2021-10-13_0006_bp3_pump_sweep_no_sig_pump_pwr_7.03_dBm_/2021-10-13_0006_bp3_pump_sweep_no_sig_pump_pwr_7.03_dBm_.ddh5'
# filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_bp3/3_state/bp3/pulsed_pump_sweep_no_sig__extra_fine_high_power/2021-10-13/2021-10-13_0035_bp3_pump_sweep_no_sig_pump_pwr_8.48_dBm_/2021-10-13_0035_bp3_pump_sweep_no_sig_pump_pwr_8.48_dBm_.ddh5'

fidelity = PU.extract_3pulse_histogram_from_filepath(filepath, 
                                            numRecords =  7686, 
                                            IQ_offset = (0,0), 
                                            plot = True, 
                                            hist_scale = 25, 
                                            fit = True,
                                            boxcar = True,
                                            bc_window = [50, 150],
                                            lpf = False, 
                                            lpf_wc = 15e6, 
                                            record_track = True)
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
