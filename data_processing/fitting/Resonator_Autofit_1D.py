# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:52:48 2021

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
from dataclasses import dataclass

@dataclass
class fit_res_sweep():
    def __init__(self, 
                 raw_filepath: str,
                 save_dir: str, 
                 ind_par_name: str,
                 vna_fname: str = 'vna_frequency', 
                 vna_phase_name: str = 'vna_phase', 
                 vna_power_name: str = 'vna_power',
                 create_file: bool = True):
        #setup files
        self.datadict, self.writer = self.make_datadict()
        
        if create_file: 
            self.writer.__enter__()

        #1D Data Extraction
        dicts = all_datadicts_from_hdf5(raw_filepath)['data']
        uvphDict = dicts.extract(self.vna_phase_name)
        uvpoDict = dicts.extract(self.vna_power_name)

        
        #get the arrays back out
        self.vna_phase = uvphDict.data_vals(self.vna_phase_name)
        self.vna_power = uvpoDict.data_vals(self.vna_power_name)

        self.vna_freqs = uvphDict.data_vals(self.vna_frequency_name)*2*np.pi
        self.ind_par = uvphDict.data_vals(ind_par_name)
        
    def default_bounds(self, QextGuess, QintGuess, f0Guess, magBackGuess):
        return ([QextGuess / 1.5, QintGuess/1.5, f0Guess*0.9, magBackGuess / 5.0, -2 * np.pi],
                [QextGuess * 1.5, QintGuess*10, f0Guess*1.1, magBackGuess * 5.0, 2 * np.pi])
    
    def initial_fit(self, f0Guess, QextGuess = 50, QintGuess = 300, magBackGuess = 0.0001, bounds_func = None, smooth = False, smooth_win = 11, phaseOffGuess = 0, debug = False, adaptive_window = False, adapt_win_size = 300e6, start_index = 0):
        print("RUNNING INITIAL FIT")
        self.autofit_starting_index = start_index
        f0Guess = f0Guess*2*np.pi
        if bounds_func == None: 
            bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
        else: 
            bounds = bounds_func(QextGuess, QintGuess, f0Guess, magBackGuess)
        filt = (self.ind_par == np.unique(self.ind_par)[start_index])

        if adaptive_window: 
            filt1 = self.vna_freqs < f0Guess + adapt_win_size*2*np.pi/2
            filt2 = self.vna_freqs > f0Guess - adapt_win_size*2*np.pi/2
            filt = filt*filt1*filt2

        init_vna_freqs = np.unique(self.vna_freqs[filt])
        init_phase_trace = self.vna_phase[filt]
        init_pow_trace = self.vna_power[filt]
        
        if debug: 
            plt.figure(1)
            plt.plot(init_vna_freqs/(2*np.pi), init_phase_trace)
            plt.title("Debug1: phase")
            plt.figure(2)
            plt.plot(init_vna_freqs/(2*np.pi), init_pow_trace)
            plt.title("Debug1: power")
            
        if smooth: 
            init_phase_trace = savgol_filter(init_phase_trace, smooth_win, 3)
            init_pow_trace = savgol_filter(init_pow_trace, smooth_win, 3)
            
        lin = 10**(init_pow_trace/20)
        
        imag = lin * np.sin(init_phase_trace)
        real = lin * np.cos(init_phase_trace)

        popt, pconv = fit(init_vna_freqs, real, imag, init_pow_trace, init_phase_trace, Qguess = (QextGuess,QintGuess), f0Guess = f0Guess, real_only = 0, bounds = bounds, magBackGuess = magBackGuess, phaseGuess = phaseOffGuess)
        
        print(f'f (Hz): {np.round(popt[2]/2/np.pi, 3)}', )
        fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
        for i in range(2):
            print(f'{fitting_params[i]}: {np.round(popt[i], 2)} +- {np.round(np.sqrt(pconv[i, i]), 3)}')
        Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
        print('Q_tot: ', round(Qtot), '\nT1 (s):', round(Qtot/popt[2]), f"Kappa: {round(popt[2]/2/np.pi/Qtot)}", )
        
        self.initial_popt = popt
        self.initial_pconv = pconv
        
        plotRes(init_vna_freqs, real, imag, init_pow_trace, init_phase_trace, popt)
        
    def save_fit(self, ind_par_val, base_popt, base_pconv): 
        
        
    def semiauto_fit(self, ind_par, vna_freqs, vna_mags, vna_phases, init_popt, debug = False, savedata = False, smooth = False, smooth_win = 11, adaptive_window = False, adapt_win_size = 300e6, fourier_filter = False, fourier_cutoff = 40, pconv_tol = 2, bounds_func = None, alt_array_scale = 1):
        print("RUNNING SEMI-AUTO FIT")
        res_freqs = np.zeros(np.size(np.unique(ind_par)))
        Qints = np.zeros(np.size(np.unique(ind_par)))
        Qexts = np.zeros(np.size(np.unique(ind_par)))
        magBacks = np.zeros(np.size(np.unique(ind_par)))
        
        init_f0 = init_popt[2]
        init_Qint = init_popt[1]
        init_Qext = init_popt[0]
        init_magBack = init_popt[3]
        for i, ind_par_val in enumerate(np.unique(ind_par)): 
            first_condn = ind_par == ind_par_val
            [first_trace_freqs, first_trace_phase, first_trace_mag] = [vna_freqs[first_condn]*2*np.pi, vna_phases[first_condn], 10**(vna_mags[first_condn]/20)]
            if smooth: 
                first_trace_phase = savgol_filter(first_trace_phase, smooth_win, 3)
                first_trace_mag = savgol_filter(first_trace_mag, smooth_win, 3)
                
            imag = first_trace_mag * np.sin(first_trace_phase)
            real = first_trace_mag * np.cos(first_trace_phase)
            if fourier_filter == True: 
                if debug: 
                    plt.figure(3)
                    plt.plot(first_trace_freqs, real)
                    plt.plot(first_trace_freqs, imag)
                    plt.title('before filter')
                imag = idct(dct(imag)[fourier_cutoff:])    
                real = idct(dct(real)[fourier_cutoff:])
                if debug: 
                    plt.figure(4)
                    plt.plot(real)
                    plt.plot(imag)
                    plt.title('after filter')
            if i >= 2: 
                if adaptive_window: 
                    filt1 = first_trace_freqs<np.average(res_freqs[i-1:i])*2*np.pi+adapt_win_size*2*np.pi/2
                    filt2 = first_trace_freqs>np.average(res_freqs[i-1:i])*2*np.pi-adapt_win_size*2*np.pi/2
                    filt= filt1*filt2
                else: 
                    filt = np.ones(np.size(first_trace_freqs)).astype(bool)
                #start averaging the previous fits for prior information to increase robustness to bad fits
                f0Guess = np.average(res_freqs[i-1:i])*2*np.pi
                magBackGuess = np.average(magBacks[i-1:i])
                (QextGuess,QintGuess) = (np.average(Qexts[i-1:i]),np.average(Qints[i-1:i]))
            else: 
                f0Guess = init_f0
                magBackGuess = init_magBack
                (QextGuess, QintGuess) = (init_Qext, init_Qint)
                filt = np.ones(np.size(first_trace_freqs)).astype(bool)
            if bounds_func == None:
                bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
            else: 
                bounds = bounds_func(QextGuess, QintGuess, f0Guess, magBackGuess)
            if i>2: 
                prev_pconv = pconv
            #fit(freq, real, imag, mag, phase, Qguess=(2e3, 1e3),real_only = 0, bounds = None)
            popt, pconv = fit(first_trace_freqs[filt], real[filt], imag[filt], first_trace_mag, first_trace_phase, Qguess = (QextGuess,QintGuess), f0Guess = f0Guess, real_only = 0, bounds = bounds, magBackGuess = magBackGuess)
            
            #catch a sudden change in convergence and try again until it's back in range: 
            if i>2: 
                pconv_diff_ratio = (np.array(pconv[0,0], pconv[1,1])-np.array(prev_pconv[0,0], prev_pconv[1,1]))/np.array(prev_pconv[0,0], prev_pconv[1,1])
                if debug: 
                    print(f"Pconv ratio: {pconv_diff_ratio}")
                j = 0
                alt_array = alt_array_scale*np.array([1e6,-1e6,5e6,-5e6, 10e6,-10e6,15e6,-15e6, 20e6, -20e6, 30e6, -30e6])*2*np.pi
                while np.any(np.abs(pconv_diff_ratio)>pconv_tol): 
                    if j>11: 
                        raise Exception("No good fit at this point")
                    print(f"sudden change in Q detected (pconv_diff_ratio: {pconv_diff_ratio}), trying resonant guess + {alt_array[j]/(2*np.pi)}")
                    #try above
                    if debug: 
                        if j%2 ==0: 
                            print("trying above")
                        else: 
                            print("trying_below")
                    popt, pconv = fit(first_trace_freqs[filt], real[filt], imag[filt], first_trace_mag, first_trace_phase, Qguess =         (QextGuess,QintGuess), f0Guess = f0Guess+alt_array[j], real_only = 0, bounds = bounds, magBackGuess = magBackGuess)
                    
                    pconv_diff_ratio = (np.array(pconv[0,0], pconv[1,1])-np.array(prev_pconv[0,0], prev_pconv[1,1]))/np.array(prev_pconv[0,0], prev_pconv[1,1])
                    j+=1
                
            
            if debug: 
                import time
                plotRes(first_trace_freqs[filt], real[filt], imag[filt], first_trace_mag[filt], first_trace_phase[filt], popt)
                time.sleep(1)
            
            res_freqs[i] = popt[2]/(2*np.pi)
            Qints[i] = popt[1]
            Qexts[i] = popt[0]
            magBacks[i] = popt[3]
            
            print(f'f (Hz): {np.round(popt[2]/2/np.pi, 3)}', )
            fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
            for i in range(2):
                print(f'{fitting_params[i]}: {np.round(popt[i], 2)} +- {np.round(np.sqrt(pconv[i, i]), 3)}')
            Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
            print('Q_tot: ', round(Qtot), '\nT1 (s):', round(Qtot/popt[2]), f"Kappa: {round(popt[2]/2/np.pi/Qtot)}", )
            if savedata:
                self.save_fit(ind_par_val, popt, pconv)
            
        return np.unique(ind_par), res_freqs, Qints, Qexts, magBacks