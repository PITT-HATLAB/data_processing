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

#%%Normalization run, to calibrate the amp gain

IQ_offset =  np.array((0,0))
records_per_pulsetype = 3840
cf = 6.151798714e9
# amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/Pump_power_sweeps/1/2021-06-30/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094/2021-06-30_0011_LO_6152798714.0_pwr_-8.69_amp_1_rotation_phase_2.094.ddh5'
amp_off_filepath = r'Z:/Data/C1/C1_Hakan/Gain_pt_0.103mA/amp_off_calibration/2021-06-30/2021-06-30_0003_LO_6152798714.0_pwr_-20_amp_0_rotation_phase_0.0/2021-06-30_0003_LO_6152798714.0_pwr_-20_amp_0_rotation_phase_0.0.ddh5'
bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(amp_off_filepath, 
                                                                                           odd_only = 0, 
                                                                                           numRecords = int(3840*2), 
                                                                                           IQ_offset = IQ_offset, 
                                                                                           plot = True)

amp_off_even_fit = PU.fit_2D_Gaussian('amp_off_even', bins_even, h_even, 
                                        guessParam[0],
                                        max_fev = 1000,
                                        contour_line = 2)
amp_off_odd_fit = PU.fit_2D_Gaussian('amp_off_odd', bins_odd, h_odd,
                                        guessParam[1],
                                        max_fev = 1000,
                                        contour_line = 2)
even_fit = amp_off_even_fit
odd_fit = amp_off_odd_fit

histogram_data_fidelity = 1-1/2*np.sum(np.sqrt((h_odd/records_per_pulsetype)*(h_even/records_per_pulsetype)))
        
bins_fine = np.linspace(np.min([bins_even, bins_odd]), np.max([bins_even, bins_odd]), 1000)

even_fit_h = PU.Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *even_fit.info_dict['popt'])/(2*np.pi*even_fit.info_dict['amplitude']*even_fit.info_dict['sigma_x']*even_fit.info_dict['sigma_y'])

odd_fit_h = PU.Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *odd_fit.info_dict['popt'])/(2*np.pi*odd_fit.info_dict['amplitude']*odd_fit.info_dict['sigma_x']*odd_fit.info_dict['sigma_y'])

fit_fidelity = 1-1/2*np.sum(np.sqrt(np.abs(even_fit_h)*np.abs(odd_fit_h)))

fig, ax = plt.subplots()
pc = ax.pcolormesh(bins_even, bins_even, h_even)
amp_off_even_fit.plot_on_ax(ax)
ax.add_patch(amp_off_even_fit.sigma_contour())
ax.set_aspect(1)
plt.colorbar(pc)

fig, ax = plt.subplots()
pc = ax.pcolormesh(bins_odd, bins_odd, h_odd)
amp_off_odd_fit.plot_on_ax(ax)
ax.add_patch(amp_off_odd_fit.sigma_contour())
ax.set_aspect(1)
plt.colorbar(pc)

amp_off_voltage = np.average([np.linalg.norm(amp_off_odd_fit.center_vec()), np.linalg.norm(amp_off_even_fit.center_vec())])*1000

#%%process everything into plottr's format so that I dont lose my mind with nested dictionaries
datadir = r'Z:\Data\C1\C1_Hakan\Gain_pt_0.103mA\Pump_power_sweeps\1'
filepaths = find_all_ddh5(datadir)
# IQ_offset =  np.array((0.8358519968365662, 0.031891495461217216))/1000
amp_on_list = [f for f in filepaths if f.find('amp_1')!=-1]
amp_off_list = [f for f in filepaths if f.find('amp_0')!=-1]
#sort everything
savedir = r'Z:\Data\C1\C1_Hakan\Gain_pt_0.103mA\Pump_power_sweeps\1_fit'
savename = 'LO_PWR_PHASE_fits'
data = dd.DataDict(
        detuning = dict(unit = 'MHz'),
        pump_power = dict(unit = 'dBm'),
        phase = dict(unit = 'dBm'),
        
        sigma_x_even = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        sigma_y_even = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        voltage_even = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        sigma_x_odd = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        sigma_y_odd = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        voltage_odd = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        sep_voltage = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV'),
        sep_over_sigma = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV/mV'),
        
        even_power_gain = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'dB'),
        odd_power_gain = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'dB'), 
        avg_power_gain =  dict(axes=['detuning', 'pump_power', 'phase'], unit = 'dB')
        )

with dds.DDH5Writer(savedir, data, name=savename) as writer:
    for filepath in amp_on_list: 
        det = (float(filepath.split('LO_')[-1].split('_')[0])-cf)/1e6
        pwr = (float(filepath.split('pwr_')[-1].split('_')[0]))
        phase = float(filepath.split('phase_')[-1].split('.ddh5')[0])
        records_per_pulsetype = 3840
        bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(filepath, 
                                                                                                   odd_only = 0, 
                                                                                                   numRecords = records_per_pulsetype*2, 
                                                                                                   IQ_offset = IQ_offset, 
                                                                                                   plot = False)
        
        even_fit = PU.fit_2D_Gaussian(f'det_{det}_pwrr_{pwr}_phase_{phase}_even', bins_even, h_even, 
                                             guessParam[0], 
                                             max_fev = 1000, 
                                             contour_line = 2)
        
        odd_fit = PU.fit_2D_Gaussian(f'det_{det}_pwrr_{pwr}_phase_{phase}_odd', bins_odd, h_odd, 
                                            guessParam[1],
                                            max_fev = 1000, 
                                            contour_line = 2)
        histogram_data_fidelity = 1-1/2*np.sum(np.sqrt((h_odd/records_per_pulsetype)*(h_even/records_per_pulsetype)))
        
        bins_fine = np.arange(np.min([bins_even, bins_odd]), np.max([bins_even, bins_odd]), 1e4)
        
        even_fit_h = PU.Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *even_fit.info_dict['popt'])/(2*np.pi*even_fit.info_dict['amplitude']*even_fit.info_dict['sigma_x']*even_fit.info_dict['sigma_y'])
        
        odd_fit_h = PU.Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *odd_fit.info_dict['popt'])/(2*np.pi*odd_fit.info_dict['amplitude']*odd_fit.info_dict['sigma_x']*odd_fit.info_dict['sigma_y'])
        
        fit_fidelity = 1-1/2*np.sum(np.sqrt(even_fit_h*odd_fit_h))

        writer.add_data(
            detuning =  det, 
            pump_power = pwr, 
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
            avg_power_gain = 20*np.log10(np.average([np.linalg.norm(even_fit.center_vec()), np.linalg.norm(odd_fit.center_vec())])*1000/amp_off_voltage)
            )