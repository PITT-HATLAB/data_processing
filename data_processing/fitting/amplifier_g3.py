# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:44:27 2021

@author: Hatlab_3
purpose: calculate qmplifier g3 from applied pump power and resonant frequency
"""

#import tools for processing taco data
import easygui
import os 
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt
from measurement_modules.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target, log_normalize_up_to_row, find_all_ddh5
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import matplotlib.colors as color
import time

from data_processing.ddh5_Plotting.TACO_multiplot_b1 import superTACO_Bars

def g3_from_pump_power(gains: np.ndarray, 
                       pump_powers: np.ndarray, 
                       mode_kappas: np.ndarray, 
                       pump_omegas: np.ndarray,
                       pump_detunings_from_res: np.ndarray
                       ): 
    '''
    Source for calculation: https://arxiv.org/abs/1605.00539
    "Introduction to Quantum-limited Parametric Amplification of Quantum Signals with Josephson Circuits"
    by Michelle Devoret and Ananda Roy
    
    Parameters
    ----------
    gains : np.ndarray
        gain in dB, whose positions correspond to the powers given in the pump_powers section
    pump_powers : np.ndarray
        pump power in dBm that the amplifier sees. This must include all attenuation in the entire line
    mode_kappas : np.ndarray
        mode kappa in 2pi*Hz
    pump_omegas : np.ndarray
        pump frequency in 2pi*Hz
    pump_detunings_from_res : np.ndarray
        pump detuning in 2pi(f-f0) where f0 is the resonator frequency in hz

    Returns
    -------
    numPumpPhotons : np.ndarray
        The sqrt of the number of pump photons expected in the pumping resonator.
    g3_arr : np.ndarray
        The third order coupling in Hz for each combination of inputs
    '''
    
    hbar = 1.0545718e-34
    lin_pump_powers = np.power(10,pump_powers/10)*0.001
    #get the expected value of pump photons present in the resonator
    numPumpPhotons = np.sqrt(8*mode_kappas*lin_pump_powers/(pump_omegas*hbar))/np.absolute(mode_kappas-2j*pump_detunings_from_res)
    Lin_Power_gains = np.power(10,gains/20)
    lpg = Lin_Power_gains
    g3_arr = -0.5*(mode_kappas/numPumpPhotons)*np.sqrt((np.sqrt(lpg)-1)/(np.sqrt(lpg)+1))
    return numPumpPhotons, g3_arr

print(g3_from_pump_power(20,-80, 2*np.pi*17e6, 2*np.pi*2*6e9, 2*np.pi*6e9))

#%%
if __name__ == '__main__':
    #make a function to convert bias currents into flux in radians
    quanta_offset = -8.98e-5
    quanta_size = 220.6e-6
    
    conv_func = lambda c: (c-quanta_offset)/quanta_size
    
    gain_cwd = r'E:\Data\Cooldown_20210408\SNAIL_Amps\C1\Best_Tacos\Gain'
    res = find_all_ddh5(gain_cwd)
    info_dict, bias_currents, best_gen_freqs, best_gen_powers, gains = superTACO_Bars(res, angles = [60,20], quanta_size = quanta_size, quanta_offset = quanta_offset, bardims = [0.001, 0.7], barbase = -24, plot = False)
    best_gen_powers[1] = best_gen_powers[1]-10
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(131)
    total_line_attenuation = 72
    ax.plot(conv_func(bias_currents), np.array(best_gen_powers)-total_line_attenuation, 'b.', markersize = 15)
    ax.set_title(r'Lowest 20dB Power (dBm) vs. Flux ($\Phi_0$)')
    ax.set_xlabel('Flux Quanta ($\Phi/\Phi_0)$')
    ax.set_ylabel('Generator Power @20dB Gain (dBm)')
    ax.grid()
    
    s = np.shape(bias_currents)
    s_arr = np.ones(s[0])
    print(s[0])
    #plotting the g3 vs flux
    ax2 = fig2.add_subplot(132)
    print(best_gen_powers-total_line_attenuation)
    num_pump_photons, g3_arr = g3_from_pump_power(gains,best_gen_powers-total_line_attenuation, 2*np.pi*30e6*s_arr, 2*np.pi*best_gen_freqs, 2*np.pi*best_gen_freqs/2)
    ax2.plot(conv_func(bias_currents), np.abs(g3_arr)/1e6, 'b.', markersize = 15)
    ax2.set_title(r'Measured g3 (MHz) vs. Flux ($\Phi_0$)')
    ax2.set_xlabel('Flux Quanta ($\Phi/\Phi_0)$')
    ax2.set_ylabel('g3 coupling (MHz)')
    ax2.grid()
    
    ax3 = fig2.add_subplot(133)
    ax3.plot(conv_func(bias_currents), num_pump_photons**2, 'b.', markersize = 15)
    ax3.set_title(r'Number of Pump Photons in res vs. Flux ($\Phi_0$)')
    ax3.set_xlabel('Flux Quanta ($\Phi/\Phi_0)$')
    ax3.set_ylabel('Number of Pump Photons')
    ax3.grid()
    
    
    # plt.vlines(conv_func(-0.173e-3), np.min(best_gen_powers-total_line_attenuation), np.max(best_gen_powers-total_line_attenuation), linestyles = 'dashed', colors = ['red'])


