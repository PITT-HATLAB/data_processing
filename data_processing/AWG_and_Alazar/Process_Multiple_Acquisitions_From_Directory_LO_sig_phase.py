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
records_per_pulsetype = 3870
cf = 6.151798714e9
# amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/Pump_power_sweeps/1/2021-06-30/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094.ddh5'
# amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/signal_power_sweeps/1_initial_guess/2021-07-06/2021-07-06_0003_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.0_V_Phase_0.0_rad_/2021-07-06_0003_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.0_V_Phase_0.0_rad_.ddh5'

filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/signal_power_sweeps/3_mid_powers/2021-07-07/2021-07-07_0043_Amp_1__LO_freq_6154298714.0_Hz_Sig_Volt_0.35_V_Phase_0.0_rad_/2021-07-07_0043_Amp_1__LO_freq_6154298714.0_Hz_Sig_Volt_0.35_V_Phase_0.0_rad_.ddh5'
# filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/signal_power_sweeps/3_mid_powers/2021-07-06/2021-07-06_0004_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.25_V_Phase_0.0_rad_/2021-07-06_0004_Amp_0__LO_freq_6153298714.0_Hz_Sig_Volt_0.25_V_Phase_0.0_rad_.ddh5'
# PU.get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False, hist_scale = 0.01, records_per_pulsetype = 3870*2)
# IQ_offset = PU.get_IQ_offset_from_filepath(amp_off_filepath, plot = False, hist_scale = 0.02, records_per_pulsetype = 3870*2)
data_fidelity, fit_fidelity = PU.get_fidelity_from_filepath(filepath, plot = True, hist_scale = 0.03, records_per_pulsetype = 3840*2)
print(data_fidelity, fit_fidelity)
IQ_offset = (0,0)

#%%process everything into plottr's format so that I dont lose my mind with nested dictionaries
datadir = r'Z:\Data\C1\C1_Hakan\Gain_pt_0.103mA\signal_power_sweeps\3_mid_powers\2021-07-07'
filepaths = find_all_ddh5(datadir)
amp_on_filepaths = [f for f in filepaths if f.lower().find('amp_1')!=-1]
amp_off_filepaths = [f for f in filepaths if f.lower().find('amp_0')!=-1]
# IQ_offset =  np.array((0.8358519968365662, 0.031891495461217216))/1000
#sort everything
savedir = r'Z:\Data\C1\C1_Hakan\Gain_pt_0.103mA\signal_power_sweeps\2_fit'
savename = 'LO_VOLT_PHASE_fits'
data = dd.DataDict(
        detuning = dict(unit = 'MHz'),
        sig_voltage = dict(unit = 'V'),
        phase = dict(unit = 'dBm'),
        
        sigma_x_even = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        sigma_y_even = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        voltage_even = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        sigma_x_odd = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        sigma_y_odd = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        voltage_odd = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        sep_voltage = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV'),
        
        sep_over_sigma = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'mV/mV'),
        even_power_gain = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'dB'),
        odd_power_gain = dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'dB'),
        avg_power_gain =  dict(axes=['detuning', 'sig_voltage', 'phase'], unit = 'dB'),
        
        histogram_fidelity =  dict(axes=['detuning', 'sig_voltage', 'phase']), 
        histogram_infidelity = dict(axes=['detuning', 'sig_voltage', 'phase']),
        
        fit_fidelity =  dict(axes=['detuning', 'sig_voltage', 'phase']), 
        fit_infidelity = dict(axes=['detuning', 'sig_voltage', 'phase'])
        )

with dds.DDH5Writer(savedir, data, name=savename) as writer:
    for i in range(len(amp_on_filepaths)):
        amp_on_filepath = amp_on_filepaths[i]
        amp_off_filepath = amp_off_filepaths[i]
        
        det = (float(amp_on_filepath.lower().split('lo_freq_')[-1].split('_')[0])-cf)/1e6
        volt = (float(amp_on_filepath.lower().split('volt_')[-1].split('_')[0]))
        phase = float(amp_on_filepath.lower().split('phase_')[-1].split('_')[0])
        records_per_pulsetype = 3840
        
        amp_off_voltage = PU.get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False)
        
        bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(amp_on_filepath, 
                                                                                                   odd_only = 0, 
                                                                                                   numRecords = records_per_pulsetype*2, 
                                                                                                   IQ_offset = IQ_offset, 
                                                                                                   plot = False)
        h_odd_norm = np.copy(h_odd/np.sum(h_odd))
        h_even_norm = np.copy(h_even/np.sum(h_even))
        
        even_fit = PU.fit_2D_Gaussian(f'det_{det}_volt_{volt}_phase_{phase}_even', bins_even, h_even, 
                                             guessParam[0], 
                                             max_fev = 1000, 
                                             contour_line = 2)
        
        odd_fit = PU.fit_2D_Gaussian(f'det_{det}_volt_{volt}_phase_{phase}_odd', bins_odd, h_odd, 
                                            guessParam[1],
                                            max_fev = 1000, 
                                            contour_line = 2)
        histogram_data_product = 1-1/2*np.sum(np.sqrt((h_odd/records_per_pulsetype)*(h_even/records_per_pulsetype)))
        
        bins_fine = np.arange(np.min([bins_even, bins_odd]), np.max([bins_even, bins_odd]), 1e4)
        
        even_fit_h = PU.Gaussian_2D(np.meshgrid(bins_even[:-1], bins_even[:-1]), *even_fit.info_dict['popt'])
        even_fit_h_norm = np.copy(even_fit_h/np.sum(even_fit_h))
        
        odd_fit_h = PU.Gaussian_2D(np.meshgrid(bins_odd[:-1], bins_odd[:-1]), *odd_fit.info_dict['popt'])
        odd_fit_h_norm = np.copy(odd_fit_h/np.sum(odd_fit_h))
        
        is_even = PU.hist_discriminant(even_fit_h, odd_fit_h)
        is_odd = np.logical_not(is_even)
        
        data_fidelity = 1-np.sum(h_odd_norm[is_even], dtype = "float64")-np.sum(h_even_norm[is_odd], dtype = "float64")
        
        fit_fidelity = 1-np.sum(odd_fit_h_norm[is_even], dtype = "float64")-np.sum(even_fit_h_norm[is_odd], dtype = "float64")
        
        writer.add_data(
            detuning =  det, 
            sig_voltage = volt, 
            phase = phase, 
            
            sigma_x_even = even_fit.info_dict['sigma_x']*1000,
            sigma_x_odd = odd_fit.info_dict['sigma_x']*1000,
            
            sigma_y_even = even_fit.info_dict['sigma_y']*1000,
            sigma_y_odd = odd_fit.info_dict['sigma_y']*1000,
            
            voltage_even = np.linalg.norm(even_fit.center_vec())*1000, 
            voltage_odd = np.linalg.norm(odd_fit.center_vec())*1000,
            
            sep_voltage = np.linalg.norm((even_fit-odd_fit).center_vec())*1000, 
            sep_over_sigma = np.linalg.norm((even_fit-odd_fit).center_vec())/np.linalg.norm(
                np.array([np.average([even_fit.info_dict['sigma_x'], odd_fit.info_dict['sigma_x']]), 
                          np.average([even_fit.info_dict['sigma_y'], odd_fit.info_dict['sigma_y']])])
                ),
            even_power_gain = 20*np.log10(np.linalg.norm(even_fit.center_vec())*1000/amp_off_voltage), 
            odd_power_gain = 20*np.log10(np.linalg.norm(odd_fit.center_vec())*1000/amp_off_voltage), 
            avg_power_gain = 20*np.log10(np.average([np.linalg.norm(even_fit.center_vec()), np.linalg.norm(odd_fit.center_vec())])*1000/amp_off_voltage),
            
            histogram_fidelity = data_fidelity, 
            histogram_infidelity = 1-data_fidelity,
            
            fit_fidelity = fit_fidelity, 
            fit_infidelity = 1-fit_fidelity
            )