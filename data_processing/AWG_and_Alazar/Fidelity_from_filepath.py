# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 17:08:39 2021
@author: Hatlab-RRK
"""

from data_processing.signal_processing.Pulse_Processing_utils import get_fidelity_from_filepath

filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/signal_power_sweeps/3_mid_powers/2021-07-06/2021-07-06_0019_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.5_V_Phase_0.0_rad_/2021-07-06_0019_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.5_V_Phase_0.0_rad_.ddh5'

data_fidelity, fit_fidelity = get_fidelity_from_filepath(filepath, plot = True, hist_scale = None, records_per_pulsetype = 3840*2)

print(data_fidelity, fit_fidelity)