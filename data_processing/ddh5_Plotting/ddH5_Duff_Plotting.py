# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:14:25 2021

@author: Hatlab_3
"""
# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target
from data_processing.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fftpack import dct, idct

# FS_filepath = r'Z:/Data/SA_2X_B1/fluxsweep/fits/2021-07-22/2021-07-22_0024_SA_2X_B1/2021-07-22_0024_SA_2X_B1.ddh5'
# Duff_filepath = r'Z:/Data/SA_2X_B1/duffing/2021-07-23/2021-07-23_0004_SA_2X_B1_duffing/2021-07-23_0004_SA_2X_B1_duffing.ddh5'
# save_filepath = r'Z:\Data\SA_2X_B1\duffing\fits'

#%%Create measurement-based saver for the fit data. 

class fit_Duff_Measurement():
    '''
    outline: 
        - Take in an existing Wolfie-format duffing file and manual fit popt
        - run semiauto_fit for each generator power, return an array of popts in an N_currents x M_gen_powers array
        - use plotRes to debug fitter
        - generate duffing graph
        
    '''
    def __init__(self, name):
        #setup files
        self.name = name
        
    def create_file(self, save_filepath): 
        self.datadict = dd.DataDict(
            current = dict(unit='A'),
            gen_power = dict(unit = 'dBm'),
            
            base_resonant_frequency = dict(axes = ['current']),
            low_power_res_frequency = dict(axes = ['current']),
            
            driven_resonant_frequency= dict(axes = ['current', 'gen_power']), 
            driven_Qint = dict(axes = ['current', 'gen_power']), 
            driven_Qext = dict(axes = ['current', 'gen_power']),
            
            driven_resonant_frequency_error= dict(axes = ['current', 'gen_power']), 
            driven_Qint_error = dict(axes = ['current', 'gen_power']), 
            driven_Qext_error = dict(axes = ['current', 'gen_power']),
            
            res_shift_ref_undriven = dict(axes = ['current', 'gen_power']), 
            res_shift_ref_low = dict(axes = ['current', 'gen_power'])
        )
        self.datadir = save_filepath
        self.writer = dds.DDH5Writer(self.datadir, self.datadict, name=self.name)
        self.writer.__enter__()
        return None
    
    def load_data(self, Duff_filepath, FS_filepath, current_filt = None): 
        #Duffing Data Extraction
        duff_dicts = all_datadicts_from_hdf5(Duff_filepath)
        duffDict = duff_dicts['data']
        uvphDict = duffDict.extract('undriven_vna_phase')
        uvpoDict = duffDict.extract('undriven_vna_power')
        dvphDict = duffDict.extract('driven_vna_phase')
        dvpoDict = duffDict.extract('driven_vna_power')
        
        if current_filt == None: 
            lower = np.min(uvphDict.data_vals('current'))
            upper = np.max(uvphDict.data_vals('current'))
        else: 
            [lower, upper] = current_filt
            #get the arrays back out
            
        filt = (uvphDict.data_vals('current')<upper)*(uvphDict.data_vals('current')>lower)
        self.undriven_vna_phase = uvphDict.data_vals('undriven_vna_phase')[filt]
        self.undriven_vna_power = uvpoDict.data_vals('undriven_vna_power')[filt]
        self.driven_vna_phase = dvphDict.data_vals('driven_vna_phase')[filt]
        self.driven_vna_power= dvpoDict.data_vals('driven_vna_power')[filt]
        self.vna_freqs = uvphDict.data_vals('vna_frequency')[filt]*2*np.pi
        self.currents = uvphDict.data_vals('current')[filt]
        self.gen_powers = dvpoDict.data_vals('gen_power')[filt]

                        
        
        self.res_func, self.qint_func, self.qext_func = self.read_fs_data(FS_filepath)
        return None
        
    def read_fs_data(self, fs_filepath, interpolation = 'linear'):
        ret = all_datadicts_from_hdf5(fs_filepath)
        res_freqs = ret['data'].extract('base_resonant_frequency').data_vals('base_resonant_frequency')
        currents = ret['data'].extract('base_resonant_frequency').data_vals('current')
        Qexts = ret['data'].extract('base_Qext').data_vals('base_Qext')
        Qints = ret['data'].extract('base_Qint').data_vals('base_Qint')
        fs_res_fit_func = interp1d(currents, res_freqs, interpolation)
        fs_Qint_fit_func = interp1d(currents, Qints, interpolation)
        fs_Qext_fit_func = interp1d(currents, Qexts, interpolation)
        return fs_res_fit_func, fs_Qint_fit_func, fs_Qext_fit_func
    
    def save_fit(self, currents, gen_power,driven_popts, driven_pconvs, low_power_res_fit_func): 
        for i, current in enumerate(np.unique(currents)): 
            driven_popt = driven_popts[i]
            driven_pconv = driven_pconvs[i]
            res_freq_ref = float(self.res_func(current))    
            res_freq_low_power = float(low_power_res_fit_func(current))
            self.writer.add_data(
                current = current, 
                gen_power = gen_power,
                
                base_resonant_frequency = res_freq_ref, 
                low_power_res_frequency = res_freq_low_power,
                
                driven_resonant_frequency = driven_popt[2]/(2*np.pi), 
                driven_Qint = driven_popt[1],
                driven_Qext = driven_popt[0],
                
                driven_resonant_frequency_error = np.sqrt(driven_pconv[2,2])/(2*np.pi),
                driven_Qint_error = np.sqrt(driven_pconv[1,1]),
                driven_Qext_error = np.sqrt(driven_pconv[0,0]),
                
                res_shift_ref_undriven = (driven_popt[2]/(2*np.pi)-res_freq_ref),
                res_shift_ref_low = (driven_popt[2]/(2*np.pi)-res_freq_low_power)
                
                )
            
    def single_fit(self, vna_freqs, phase_trace, pow_trace, f0Guess, QextGuess = 50, QintGuess = 300, magBackGuess = 0.0001, bounds = None, smooth = False, smooth_win = 11, phaseOffGuess = 0, debug = False, adaptive_window = False, adapt_win_size = 300e6): 
        f0Guess = f0Guess*2*np.pi
        if bounds == None: 
            bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
            
        if adaptive_window: 
            filt1 = vna_freqs < f0Guess + adapt_win_size*2*np.pi
            filt2 = vna_freqs > f0Guess - adapt_win_size*2*np.pi
            filt = filt1*filt2
            vna_freqs = np.copy(vna_freqs[filt])
            
        if debug: 
            plt.figure(1)
            plt.plot(vna_freqs/(2*np.pi), phase_trace)
            plt.title("Debug1: phase")
            plt.figure(2)
            plt.plot(vna_freqs/(2*np.pi), pow_trace)
            plt.title("Debug1: power")
            
        if smooth: 
            phase_trace = savgol_filter(phase_trace, smooth_win, 3)
            pow_trace = savgol_filter(pow_trace, smooth_win, 3)
            
        lin = 10**(pow_trace/20)
        
        imag = lin * np.sin(phase_trace)
        real = lin * np.cos(phase_trace)

        popt, pconv = fit(vna_freqs, real, imag, pow_trace, phase_trace, Qguess = (QextGuess,QintGuess), f0Guess = f0Guess, real_only = 0, bounds = bounds, magBackGuess = magBackGuess, phaseGuess = phaseOffGuess)
        
        print(f'f (Hz): {np.round(popt[2]/2/np.pi, 3)}', )
        fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
        for i in range(2):
            print(f'{fitting_params[i]}: {np.round(popt[i], 2)} +- {np.round(np.sqrt(pconv[i, i]), 3)}')
        Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
        print('Q_tot: ', round(Qtot), '\nT1 (s):', round(Qtot/popt[2]), f"Kappa: {round(popt[2]/2/np.pi/Qtot)}", )
        
        self.initial_popt = popt
        self.initial_pconv = pconv
        
        if debug: 
            plotRes(vna_freqs, real, imag, pow_trace, phase_trace, popt)
            
        return(popt, pconv)
    
    def initial_fit(self, f0Guess, QextGuess = 50, QintGuess = 300, magBackGuess = 0.0001, bounds = None, smooth = False, smooth_win = 11, phaseOffGuess = 0, debug = False, adaptive_window = False, adapt_win_size = 300e6):
        if bounds == None: 
            bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
            
        filt = (self.currents == np.unique(self.currents)[0])*(self.gen_powers == np.unique(self.gen_powers)[0])
        print(np.size(filt))

        init_vna_freqs = np.unique(self.vna_freqs[filt])
        print(np.size(init_vna_freqs))
        init_phase_trace = self.undriven_vna_phase[filt]
        print(np.size(init_phase_trace))
        init_pow_trace = self.undriven_vna_power[filt]
        print(np.size(init_pow_trace))
        
        if debug: 
            plt.figure(1)
            plt.plot(init_vna_freqs/(2*np.pi), init_phase_trace)
            plt.title("Debug1: phase")
            plt.figure(2)
            plt.plot(init_vna_freqs/(2*np.pi), init_pow_trace)
            plt.title("Debug1: power")
            
        lin = 10**(init_pow_trace/20)
        
        imag = lin * np.sin(init_phase_trace)
        real = lin * np.cos(init_phase_trace)

        popt, pconv = self.single_fit(init_vna_freqs, init_phase_trace, init_pow_trace, f0Guess, QintGuess = QintGuess, QextGuess = QextGuess, magBackGuess = magBackGuess, phaseOffGuess = phaseOffGuess, adaptive_window = adaptive_window, adapt_win_size = adapt_win_size, debug = debug)
        self.initial_popt = popt
        self.initial_pconv = pconv
        
        plotRes(init_vna_freqs, real, imag, init_pow_trace, init_phase_trace, popt)
        
    def default_bounds(self, QextGuess, QintGuess, f0Guess, magBackGuess):
        return ([QextGuess / 1.5, QintGuess / 1.5, f0Guess/2, magBackGuess / 2, -2*np.pi],
                [QextGuess * 1.5, QintGuess +200, f0Guess*2, magBackGuess * 2, 2*np.pi])
    
    def semiauto_fit(self, bias_currents, vna_freqs, vna_mags, vna_phases, popt, 
                     debug = False, 
                     smooth = False, 
                     smooth_win = 11, 
                     adaptive_window = False, 
                     adapt_win_size = 300e6, 
                     fourier_filter = False, 
                     fourier_cutoff = 40, 
                     pconv_tol = 10, 
                     bounds = None):
        res_freqs = np.zeros(np.size(np.unique(bias_currents)))
        Qints = np.zeros(np.size(np.unique(bias_currents)))
        Qexts = np.zeros(np.size(np.unique(bias_currents)))
        magBacks = np.zeros(np.size(np.unique(bias_currents)))
        popts = []
        pconvs = []
        
        init_f0 = popt[2]
        init_Qint = popt[1]
        init_Qext = popt[0]
        init_magBack = popt[3]
        
        for i, current in enumerate(np.sort(np.unique(bias_currents))): 
            first_condn = bias_currents == current
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
            if bounds == None: 
                bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
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
                alt_array = np.array([1e6,-1e6,5e6,-5e6, 10e6,-10e6,15e6,-15e6, 20e6, -20e6, 30e6, -30e6, 50e6, -50e6, 100e6, -100e6])*2*np.pi
                while np.any(np.abs(pconv_diff_ratio)>pconv_tol): 
                    if j>np.size(alt_array)-1: 
                        raise Exception(f"No good fit at this point: (Bias: {current}, Power: {self.latest_power})")
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
            popts.append(popt)
            pconvs.append(pconv)
            
            print(f'f (Hz): {np.round(popt[2]/2/np.pi, 3)}', )
            fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
            for i in range(2):
                print(f'{fitting_params[i]}: {np.round(popt[i], 2)} +- {np.round(np.sqrt(pconv[i, i]), 3)}')
            Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
            print('Q_tot: ', round(Qtot), '\nT1 (s):', round(Qtot/popt[2]), f"Kappa: {round(popt[2]/2/np.pi/Qtot)}", )
            
        return np.unique(bias_currents), res_freqs, Qints, Qexts, magBacks, popts, pconvs
                                          
    def fit(self, 
            debug = False, 
            save_data = False, 
            max_gen_power = None, 
            savedata = False, 
            smooth = False, 
            smooth_win = 11, 
            adaptive_window = False, 
            adapt_win_size = 300e6,  
            bounds = None,
            fourier_filter = False, 
            fourier_cutoff = 40, 
            pconv_tol = 10):
        
        if max_gen_power != None:     
            fitted_gen_powers = np.unique(self.gen_powers) <= max_gen_power
        else: 
            fitted_gen_powers = np.unique(self.gen_powers) <= np.max(np.unique(self.gen_powers))
            
            
        for i, gen_power in enumerate(np.unique(self.gen_powers)[fitted_gen_powers]): 
            self.latest_power = gen_power
            pow_condn = self.gen_powers == gen_power
            
            bias_currents = self.currents[pow_condn]
            vna_freqs = self.vna_freqs[pow_condn]
            vna_phases = self.driven_vna_phase[pow_condn]
            vna_mags = self.driven_vna_power[pow_condn]
            
            fit_currents, fit_freqs, fit_Qints, fit_Qexts, fit_magBacks, popts, pconvs = self.semiauto_fit(bias_currents, 
                                                                                                           vna_freqs/(2*np.pi), 
                                                                                                           vna_mags, 
                                                                                                           vna_phases, 
                                                                                                           self.initial_popt, 
                                                                                                           debug = debug, 
                                                                                                           smooth = smooth, 
                                                                                                           smooth_win = smooth_win, 
                                                                                                           adaptive_window = adaptive_window, 
                                                                                                           adapt_win_size = adapt_win_size, 
                                                                                                           fourier_filter = fourier_filter, 
                                                                                                           fourier_cutoff = fourier_cutoff, 
                                                                                                           pconv_tol = pconv_tol, 
                                                                                                           bounds = bounds)
            if i == 0: 
                self.low_power_res_fit_func = interp1d(fit_currents, fit_freqs, 'linear')
            if save_data: 
                self.save_fit(bias_currents, gen_power, popts, pconvs, self.low_power_res_fit_func)
    

#%%

# #Duffing Autoplot
# #main(Duff_filepath, 'data')


# FS_filepath = r'Z:/Data/SA_2X_B1/fluxsweep/fits/2021-07-22/2021-07-22_0024_SA_2X_B1/2021-07-22_0024_SA_2X_B1.ddh5'
# Duff_filepath = r'Z:/Data/SA_2X_B1/duffing/2021-07-23/2021-07-23_0010_SA_2X_B1_duffing_fine/2021-07-23_0010_SA_2X_B1_duffing_fine.ddh5'
# save_filepath = r'Z:\Data\SA_2X_B1\duffing\fits'

# DFit = fit_Duff_Measurement(Duff_filepath, FS_filepath, save_filepath, 'SA_B1_Duff_fine')
# #%%
# DFit.initial_fit(8.0e9, 
#                   QextGuess = 50, 
#                   QintGuess = 1000, 
#                   magBackGuess = 0.01, 
#                   bounds = None, 
#                   smooth = False, 
#                   smooth_win = 11,
#                   phaseOffGuess = 0, 
#                   debug = False, 
#                   adaptive_window = False, 
#                   adapt_win_size = 300e6
#                 )
# #%%
# print(np.min(DFit.gen_powers))
# #%%
# DFit.fit(
#         debug = False, 
#         save_data = True, 
#         max_gen_power = -20, 
#         savedata = True, 
#         smooth = False, 
#         smooth_win = 11, 
#         adaptive_window = True,  
#         adapt_win_size = 400e6,  
#         fourier_filter = False, 
#         fourier_cutoff = 40, 
#         pconv_tol = 10)
# #%%

    
    
    
    
    
    
    
    
    
    
    