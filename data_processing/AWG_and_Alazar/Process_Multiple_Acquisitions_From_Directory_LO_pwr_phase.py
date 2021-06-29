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

#%%
datadir = r'Z:\Data\C1\C1_Hakan\phase_preserving_checks\20dB\Multi_LO_Multi_power'
filepaths = find_all_ddh5(datadir)
# IQ_offset =  np.array((0.8358519968365662, 0.031891495461217216))/1000
IQ_offset =  np.array((0,0))
cf = 6.19665e9
#%%process everything into plottr's format so that I dont lose my mind with nested dictionaries
import collections
amp_on_list = [f for f in filepaths if f.find('amp_1')!=-1]
amp_off_list = [f for f in filepaths if f.find('amp_0')!=-1]
#sort everything
savedir = r'Z:\Data\C1\C1_Hakan\phase_preserving_checks\20dB\Multi_LO_Multi_power\fits'
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
        sep_over_sigma = dict(axes=['detuning', 'pump_power', 'phase'], unit = 'mV')
        )

with dds.DDH5Writer(savedir, data, name=savename) as writer:
    for filepath in amp_on_list: 
        det = (float(filepath.split('LO_')[-1].split('_')[0])-cf)/1e6
        pwr = (float(filepath.split('pwr_')[-1].split('_')[0]))
        phase = float(filepath.split('phase_')[-1].split('.ddh5')[0])
        
        bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(filepath, odd_only = 0, numRecords = 3840*2, IQ_offset = IQ_offset, plot = False)
        even_fit = PU.fit_2D_Gaussian(f'det_{det}_pwrr_{pwr}_phase_{phase}_even', bins_even, h_even, 
                                             guessParam[0], 
                                             max_fev = 1000, 
                                             contour_line = 2)
        
        odd_fit = PU.fit_2D_Gaussian(f'det_{det}_pwrr_{pwr}_phase_{phase}_odd', bins_odd, h_odd, 
                                            guessParam[1],
                                            max_fev = 1000, 
                                            contour_line = 2)

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
                )
            
            )

# filepaths = [r'Z:/Data/C1/C1_Hakan/phase_preserving_checks/20dB/Multi-LO_sweep/2021-06-24/2021-06-24_0001_LO_6181650000.0_20dB_Gain_pt_amp_0_rotation_phase_0.0/2021-06-24_0001_LO_6181650000.0_20dB_Gain_pt_amp_0_rotation_phase_0.0.ddh5']
#%%process by detuning
sweep_info_dict = collections.defaultdict(dict)
det_arr = np.array(list(sweep_filepath_dict.keys())).astype(float)
pwr_arr = np.array(list(sweep_filepath_dict.keys())).astype(float)
for det in det_arr:
    even_fits = []
    odd_fits = []
    names = []
    
    for phase, filepath in sweep_filepath_dict[f'{det}'].items():
        # print(phase, filepath)
        name = f"Detuning_{det}_MHz{phase}"
        names.append(name)
        print(f'processing {name}')
        bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(filepath, odd_only = 0, numRecords = 3840*2, IQ_offset = IQ_offset, plot = False)
        even_fits.append(PU.fit_2D_Gaussian(name+'_even', bins_even, h_even, 
                                             guessParam[0], 
                                             max_fev = 1000, 
                                             contour_line = 2))
        
        odd_fits.append(PU.fit_2D_Gaussian(name+'_odd', bins_odd, h_odd, 
                                            guessParam[1],
                                            max_fev = 1000, 
                                            contour_line = 2))

    fig = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    for i, fit in enumerate(even_fits+odd_fits): 
        # print(fit.info_dict['contour'][0])
        try: 
            ax.plot(fit.info_dict['contour'][0], fit.info_dict['contour'][1], label = names[i])
        except: 
            ax.plot(fit.info_dict['contour'][0], fit.info_dict['contour'][1], label = names[i//2])
        fit.plot_on_ax(ax, color = 'black')
        ax.legend()
    
    # get average sigmas and other stats
    fig, ax = plt.subplots()
    sx_arr = []
    sy_arr = []
    center_loc_arr = []
    mag_arr = []
    
    for i, fit in enumerate(even_fits+odd_fits): 
        sx_arr.append(fit.info_dict['sigma_x'])
        sy_arr.append(fit.info_dict['sigma_y'])
        center_loc_arr.append(fit.center_vec())
        mag_arr.append(np.linalg.norm(fit.center_vec()))
    
    sep_arr = []
    #get the seperation distance voltage for each
    for i in range(len(even_fits)):
        diff_class = even_fits[i]-odd_fits[i]
        sep_arr.append(np.linalg.norm(diff_class.center_vec()))
        
    
    # print("sigma x average: ", np.average(sx_arr)*1000, "mV std dev: ", np.std(sx_arr)*1000, "mV")
    # print("sigma y average: ", np.average(sy_arr)*1000, "mV std dev: ", np.std(sy_arr)*1000, "mV")
    # print("Avg magnitude: ", np.average(mag_arr))
    
    sweep_info_dict[f'{det}']['sigma_x_average_(mV)'] = np.average(sx_arr)*1000
    sweep_info_dict[f'{det}']['sigma_y_average (mV)'] = np.average(sy_arr)*1000
    sweep_info_dict[f'{det}']['avg_magnitude (mV)'] = np.average(mag_arr)*1000
    sweep_info_dict[f'{det}']['average seperation voltage (mV)'] = np.average(sep_arr)*1000
    
    
    
    
#%% plotting summary information
#extract detunings array, average_mag_array, sigma_x_array, and sigma_y array
det_arr = []
avg_mag_arr = []
sx_arr = []
sy_arr = []
for det in sweep_info_dict.keys():
    det_arr.append(float(det))
    sx_arr.append(sweep_info_dict[det]['sigma_x_average_(mV)'])
    sy_arr.append(sweep_info_dict[det]['sigma_y_average (mV)'])
    avg_mag_arr.append(sweep_info_dict[det]['avg_magnitude (mV)'])
det_arr = np.array(det_arr)
avg_mag_array = np.array(avg_mag_arr)
sx_arr = np.array(sx_arr)
sy_arr = np.array(sy_arr)


SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(det_arr, avg_mag_arr, '.')
ax.set_xlabel('detuning (MHz)')
ax.set_ylabel('Average voltage (mV)')
ax.grid()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(det_arr, avg_mag_arr/sx_arr, '.', label = '$\sigma_x$')
ax.plot(det_arr, avg_mag_arr/sy_arr, '.', label = '$\sigma_y$')
ax.set_xlabel('detuning (MHz)')
ax.set_ylabel(r'$\frac{Average\ voltage\ (mV)}{Average\ \sigma\ (mV)}$')
ax.grid()
ax.legend()










