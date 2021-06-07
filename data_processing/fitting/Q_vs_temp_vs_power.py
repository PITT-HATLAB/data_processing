# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:08:53 2021

@author: Hatlab_3
"""


import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from measurement_modules.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target, find_all_ddh5
from data_processing.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.fftpack import dct, idct

from data_processing.fitting.Resonator_Autofit_1D import fit_res_sweep as FRS

%matplotlib inline

raw_filepath = r'Z:\Texas\Cooldown_20210525\PC_CuCollar\2021-06-03\2021-06-03_0013_vna_trace_vs_vna_power_0dBatten_330mK\2021-06-03_0013_vna_trace_vs_vna_power_0dBatten_330mK.ddh5'
ind_par_name = 'Gen_power'
# name = "PC_HPAl_etched_3_Q_vs_power_fit_-83_to-20_330mK"
name = "PC_CuCollar_Q_vs_power_fit_-83_to-20_330mK"
datadict = dd.DataDict(
            vna_power = dict(unit='dBm'),
            
            base_resonant_frequency = dict(axes = ['vna_power']),
            base_Qint = dict(axes = ['vna_power']),
            base_Qext = dict(axes = ['vna_power']),
            base_Qtot = dict(axes = ['vna_power']),
            
            base_resonant_frequency_error = dict(axes = ['vna_power']),
            base_Qint_error = dict(axes = ['vna_power']),
            base_Qext_error = dict(axes = ['vna_power']),
        )
# datadir = r'Z:\Texas\Cooldown_20210525\PC_HPAl_etch_3'
datadir = r'Z:\Texas\Cooldown_20210525\PC_CuCollar'
writer = dds.DDH5Writer(datadir, datadict, name=name)
def ext_save_fit(writer, ind_par_val, base_popt, base_pconv): 
    writer.add_data(
        vna_power = ind_par_val, 
        
        base_resonant_frequency = base_popt[2]/(2*np.pi), 
        base_Qint = base_popt[1], 
        base_Qext = base_popt[0], 
        base_Qtot = 1/(1/base_popt[1]+1/base_popt[0]),
        
        base_resonant_frequency_error = np.sqrt(base_pconv[2, 2])/(2*np.pi), 
        base_Qint_error = np.sqrt(base_pconv[1, 1]), 
        base_Qext_error = np.sqrt(base_pconv[0, 0]), 
        )

FR = fit_res_sweep(datadict, writer, ext_save_fit, raw_filepath, ind_par_name, create_file = False)
#%%
# FR.initial_fit(7.4362155e9, QextGuess = 11e6, QintGuess=30e6, magBackGuess = 0.2, phaseOffGuess = 0, debug = True, smooth = False, smooth_win = 5, adaptive_window = True, adapt_win_size = 1e4)
FR.initial_fit(7499216935, QextGuess = 0.8e6, QintGuess=0.01e6, magBackGuess = 0.7, phaseOffGuess = np.pi/2, debug = True, smooth = False, smooth_win = 5, adaptive_window = True, adapt_win_size = 5e6)
#%% Automatic Fitting (be sure initial fit is good!)
gen_powers, res_freqs, Qints, Qexts, magBacks = FR.semiauto_fit(FR.ind_par, FR.vna_freqs/(2*np.pi), FR.vna_power, FR.vna_phase, FR.initial_popt, debug = True, savedata = False, smooth = False, smooth_win = 5, adaptive_window = False, adapt_win_size = 3e6, fourier_filter = False, pconv_tol = 1)
plt.figure()
plt.plot(gen_powers, Qints)
plt.figure()
plt.plot(gen_powers, Qexts)
plt.figure()
plt.plot(gen_powers, res_freqs)