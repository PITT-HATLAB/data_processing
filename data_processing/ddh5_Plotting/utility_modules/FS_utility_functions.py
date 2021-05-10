import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from hat_utilities.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target
from hat_utilities.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.fftpack import dct, idct

class fit_fluxsweep():
    def __init__(self, Flux_filepath, save_filepath, name):
        #setup files
        self.name = name
        self.datadict = dd.DataDict(
            current = dict(unit='A'),
            
            base_resonant_frequency = dict(axes = ['current']),
            base_Qint = dict(axes = ['current']),
            base_Qext = dict(axes = ['current']),
            
            base_resonant_frequency_error = dict(axes = ['current']),
            base_Qint_error = dict(axes = ['current']),
            base_Qext_error = dict(axes = ['current']),
        )
        self.datadir = save_filepath
        self.writer = dds.DDH5Writer(self.datadir, self.datadict, name=self.name)
        self.writer.__enter__()
        
        #Duffing/FS Data Extraction
        duff_dicts = all_datadicts_from_hdf5(Flux_filepath)
        duffDict = duff_dicts['data']
        uvphDict = duffDict.extract('phase')
        uvpoDict = duffDict.extract('power')

        
        #get the arrays back out
        self.undriven_vna_phase = uvphDict.data_vals('phase')
        self.undriven_vna_power = uvpoDict.data_vals('power')

        self.vna_freqs = uvphDict.data_vals('frequency')*2*np.pi
        self.currents = uvphDict.data_vals('current')
        
    def default_bounds(self, QextGuess, QintGuess, f0Guess, magBackGuess):
        return ([QextGuess / 1.5, QintGuess / 1.5, f0Guess/2, magBackGuess / 5.0, -2 * np.pi],
                [QextGuess * 1.5, QintGuess +200, f0Guess*1.5, magBackGuess * 5.0, 2 * np.pi])
    
    def initial_fit(self, f0Guess, QextGuess = 50, QintGuess = 300, magBackGuess = 0.0001, bounds = None, smooth = False, smooth_win = 11, phaseOffGuess = 0, debug = False, adaptive_window = False, adapt_win_size = 300e6):
        f0Guess = f0Guess*2*np.pi
        if bounds == None: 
            bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
            
        filt = (self.currents == np.unique(self.currents)[0])

        if adaptive_window: 
            filt1 = self.vna_freqs < f0Guess + adapt_win_size*2*np.pi/2
            filt2 = self.vna_freqs > f0Guess - adapt_win_size*2*np.pi/2
            filt = filt*filt1*filt2

        init_vna_freqs = np.unique(self.vna_freqs[filt])
        init_phase_trace = self.undriven_vna_phase[filt]
        init_pow_trace = self.undriven_vna_power[filt]
        
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
        
    def save_fit(self, current, base_popt, base_pconv): 
        self.writer.add_data(
            current = current, 
            
            base_resonant_frequency = base_popt[2]/(2*np.pi), 
            base_Qint = base_popt[1], 
            base_Qext = base_popt[0], 
            
            base_resonant_frequency_error = base_pconv[2, 2]/(2*np.pi), 
            base_Qint_error = base_pconv[1, 1], 
            base_Qext_error = base_pconv[0, 0], 
            )
    def semiauto_fit(self, bias_currents, vna_freqs, vna_mags, vna_phases, popt, debug = False, savedata = False, smooth = False, smooth_win = 11, adaptive_window = False, adapt_win_size = 300e6, fourier_filter = False, fourier_cutoff = 40, pconv_tol = 2):
        res_freqs = np.zeros(np.size(np.unique(bias_currents)))
        Qints = np.zeros(np.size(np.unique(bias_currents)))
        Qexts = np.zeros(np.size(np.unique(bias_currents)))
        magBacks = np.zeros(np.size(np.unique(bias_currents)))
        
        init_f0 = popt[2]
        init_Qint = popt[1]
        init_Qext = popt[0]
        init_magBack = popt[3]
        for i, current in enumerate(np.unique(bias_currents)): 
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
                alt_array = np.array([1e6,-1e6,5e6,-5e6, 10e6,-10e6,15e6,-15e6, 20e6, -20e6, 30e6, -30e6])*2*np.pi
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
                self.save_fit(current, popt, pconv)
            
        return np.unique(bias_currents), res_freqs, Qints, Qexts, magBacks