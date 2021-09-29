# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:43:45 2021

@author: Hatlab_3
"""
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from data_processing.models.SNAIL_supporting_modules.Participation_and_Alpha_Fitter import slider_fit
from data_processing.fitting.QFit import fit, plotRes
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from data_processing.Helper_Functions import find_all_ddh5
from data_processing.ddh5_Plotting.TACO_multiplot_b1 import superTACO_Bars
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def find_quanta(currents, res_freqs, smooth_window = 0):
    if smooth_window != 0: 
        res_freqs = savgol_filter(res_freqs, smooth_window, 2)
    max_res_current = currents[np.argmax(res_freqs)]
    min_res_current = currents[np.argmin(res_freqs)]
    quanta_size = 2*np.abs(min_res_current - max_res_current)
    quanta_offset = max_res_current
    current_to_quanta_conversion_function = lambda c: (c-quanta_offset)/quanta_size
    quanta_to_current_function = lambda q: q*quanta_size+quanta_offset
    
    return quanta_size, quanta_offset, current_to_quanta_conversion_function, quanta_to_current_function

def parallel(v1, v2): 
    return 1/(1/v1+1/v2)

def get_phi_min_funcs(alpha, phi_ext_arr): 
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail_norm = -a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3)
    c1 = sp.series(U_snail_norm, phi_s, x0 = phi_m, n = 2).removeO().coeff((phi_s-phi_m))
    #generate a lambda function that outputs another lambda function for a given phi_ext
    #which then depends on phi_m only
    func_arr = []
    for phi_ext in phi_ext_arr: 
        c1_num = sp.lambdify(phi_m, c1.subs(a, alpha).subs(phi_e, phi_ext), "numpy")
        func_arr.append(c1_num)
    return func_arr
def get_phi_min_fsolve(alpha, phi_ext_arr):
    funcs = get_phi_min_funcs(alpha, phi_ext_arr)
    sol_arr = np.ones(np.size(funcs))
    for i, func in enumerate(funcs): 
        sol_arr[i] = fsolve(func, phi_ext_arr[i])
    return sol_arr
def get_phi_min(alpha, phi_ext): 
    func = get_phi_min_funcs(alpha, [phi_ext])[0]
    return(fsolve(func, phi_ext)[0])

def c4_func_gen_vectorize(alpha_val): #can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 5)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 4))*24
    c4exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c4exp)

def c3_func_gen_vectorize(alpha_val): #can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 4)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 3))*6
    c3exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c3exp)

def c2_func_gen_vectorize(alpha_val): 
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 3)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 2))*2
    c2exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c2exp)


class SnailAmp(): 
    def __init__(self): #uA/um^2
        '''
        Parameters
        ----------
        junction_sizes : tuple
            (small_size, large_size) in micrometers squared
        quanta_start : float
            0-flux point in Amps 
        quanta_size : float
            quanta ize in Amps

        Returns
        -------
        None.
        '''
        
        self.hbar = 1.0545718e-34
        self.e = 1.60218e-19
        self.phi0 = 2*np.pi*self.hbar/(2*self.e)
        
    def generate_quanta_function(self, quanta_offset, quanta_size): 
        #function for converting bias currents to quanta fractions
        self.quanta_offset = quanta_offset
        self.quanta_size = quanta_size
        self.conv_func = lambda c: (c-quanta_offset)/quanta_size
        
    def info_from_junction_sizes(self, junction_sizes, res = 100, Jc = 0.8, verbose = False):
        
        self.s_size, self.l_size = junction_sizes
        
        self.alpha_from_sizes = self.s_size/self.l_size
        self.I0s, self.I0l = Jc*self.s_size*1e-6, Jc*self.l_size*1e-6
        
        self.Lss, self.Lsl = self.Ic_to_Lj(self.I0s), self.Ic_to_Lj(self.I0l)
        self.Ejs, self.Ejl = self.Ic_to_Ej(self.I0s), self.Ic_to_Ej(self.I0l)
        
        self.Ls0 = parallel(self.Lss, self.Lsl)
        
        self.c2_func, self.c3_func, self.c4_func = self.generate_coefficient_functions(self.alpha_from_sizes, res = res, verbose = False)
        
        return self.c2_func, self.c3_func, self.c4_func
    
    def info_from_junction_i0(self, junction_i0_small, junction_i0_large, res = 100, Jc = 0.8, verbose = False):
        '''
        junction_i0_small: junction critical current in A
        junction_i0_large: junction critical current in A
        '''
        
        self.I0s, self.I0l = junction_i0_small, junction_i0_large
        
        self.Lss, self.Lsl = self.Ic_to_Lj(self.I0s), self.Ic_to_Lj(self.I0l)
        self.Ejs, self.Ejl = self.Ic_to_Ej(self.I0s), self.Ic_to_Ej(self.I0l)
        
        self.alpha_from_i0 = self.Ejs/self.Ejl

        self.c2_func, self.c3_func, self.c4_func = self.generate_coefficient_functions(self.alpha_from_i0, res = res, verbose = False)
        
        return self.c2_func, self.c3_func, self.c4_func

    def Ic_to_Ej(self, Ic: float):
        '''
        Parameters
        ----------
        Ic : float
            critical current in amps
        Returns
        -------
        Ej in Joules
        src: https://en.wikipedia.org/wiki/Josephson_effect
        '''
        return Ic*self.phi0/(2*np.pi)
    
    def Ic_to_Lj(self, Ic: float): 
        '''
        Parameters
        ----------
        Ic : float
            critical current in amps
        Returns
        -------
        Lj in Henries
        src: https://en.wikipedia.org/wiki/Josephson_effect
        '''
        return self.phi0/(2*np.pi*Ic)
    
    def generate_coefficient_functions(self, alpha_val, res = int(100), plot = False, show_coefficients = False, verbose = False):
        '''
        Parameters
        ----------
        alpha_val : float
            alpha value between 0 and 0.33
        res : int, optional
            number of points to base interpolation off of. The default is 100.
        Returns
        -------
        c2_func : lambda function
            function that will return the value of c2
        c3_func : lambda function
            DESCRIPTION.
        c4_func : lambda function
            DESCRIPTION.

        '''
        if verbose:
            print("Calculating expansion coefficients")
        start_time = timer()
        
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c4_arr = c4_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose: 
            print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c4_func = interp1d(phi_ext_arr, c4_arr, 'quadratic')
        
        
        #c3: 
        start_time = timer()
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c3_arr = c3_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose: 
            print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c3_func = interp1d(phi_ext_arr, c3_arr, 'quadratic')
        
        
        #c2: 
        start_time = timer()
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c2_arr = c2_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        if verbose: 
            print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c2_func = interp1d(phi_ext_arr, c2_arr, 'quadratic')
        
        if plot: 
            plt.plot(phi_ext_arr, self.c2_func(phi_ext_arr), label = "c2")
            plt.plot(phi_ext_arr, self.c3_func(phi_ext_arr), label = "c3")
            plt.plot(phi_ext_arr, self.c4_func(phi_ext_arr), label = 'c4')
            
            plt.legend()
        
        return c2_func, c3_func, c4_func
        
    def gradient_descent_participation_fitter(self, fitted_res_func, initial_p_guess, initial_alpha_guess, init_f0_guess, res = 100, bounds = None):
        
        '''
        Parameters
        ----------
        fitted_res_func : function:ndarray->ndarray
            function which takes in flux fraction in [0, 1] and produces the resonant frequency of the experimental device
        
        initial_p_guess: float
            guess for the participation ratio of the SNAIL at 0-flux
        
        initial_alpha_guess: float
            guess for the ratio of large junction inductance to small junciton inductance of the SNAIL
        
        kwargs: 
            res - the number of points with which to do the fitting. Fewer is faster, more is better
        Returns
        -------
        fitted alpha
        fitted p
        '''
        fit_quanta = np.sort(np.append(np.append(np.linspace(0,1, int(res/4)), np.linspace(0.25,0.75, int(res/4))),np.linspace(0.45,0.55, int(res/2))))*2*np.pi
        
        def fit_func(quanta_arr, alpha, p_rat, f0): 
            
            print(f"P: {p_rat}, alpha: {alpha}, f0: {f0}")
            #make the c2 we need from the supplied alpha
            c2_func = c2_func_gen_vectorize(alpha)
            res_freqs = f0/(np.sqrt(1+p_rat/c2_func(quanta_arr).astype(float)))
            
            return res_freqs
        
        #fit the data
        if bounds ==None: 
            bounds = [[0.1, 0.001, fitted_res_func(0)*0.7],
                      [0.33, 1, fitted_res_func(0)*1.3]]
            
        popt, pcov = curve_fit(fit_func, fit_quanta, fitted_res_func(fit_quanta), p0 = [initial_alpha_guess, initial_p_guess, init_f0_guess], 
                               bounds = bounds)
        
        [fitted_alpha, fitted_p, fitted_f0] = popt
        [d_alpha, d_p, d_f0] = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]
        return fit_func, [fitted_alpha, fitted_p, fitted_f0], [d_alpha, d_p, d_f0]
        
    def frattini_p_to_part(self, fp, alpha):
        return 1/(1/fp*c2_func_gen_vectorize(alpha)(0)+1)
    
    def slider_participation_fitter(self, stored_fits_filepath: str, fluxsweep_filepath: str, ret_sliders = False, start_freq = 7e9): 
        '''
        Parameters
        ----------
        stored_fits_filepath : str
            path to a pickled fit file
        fluxsweep_filepath : str
            path to a fluxsweep stored in plottr's datadict format'

        Returns
        -------
        4x matplotlib.widgets.slider objects, call slider.val to get value

        '''
        self.p_arr = np.linspace(0.01, 0.3, 50)
        self.alpha_arr = np.linspace(0.1, 0.32, 50)
        #the below function returns the slider fit, which you then have to call .val on
        self.p_slider, self.a_slider, self.f_slider = slider_fit(fluxsweep_filepath, 
                                                                 stored_fits_filepath, 
                                                                 self.quanta_offset, 
                                                                 self.quanta_size, 
                                                                 self.p_arr, 
                                                                 self.alpha_arr, 
                                                                 start_freq = start_freq)
        if ret_sliders: 
            return self.p_arr, self.alpha_arr, self.p_slider, self.a_slider, self.f_slider
        else: 
            pass
    
    def vals_from_sliders(self): 
        '''
        A supporting function to slider_participation_fitter for extracting
        the alpha and p values after the sliders have been used to fit

        '''
        self.alpha_from_FS = self.alpha_arr[self.a_slider.val]
        self.p_from_FS = self.p_arr[self.p_slider.val]
        
        return self.alpha_from_FS, self.p_from_FS, self.f_slider.val
        

    
    def set_linear_inductance(self, L0): 
        self.L0 = L0
    def set_linear_capacitance(self, C0): 
        self.C0 = C0
        
    def generate_participation_function(self, L0, Lfunc): 
        return lambda phi: Lfunc(phi)/(L0+Lfunc(phi))
        
    def generate_inductance_function(self, L_large, c2_func):
        return lambda phi: L_large/c2_func(phi)
    
    def generate_resonance_function_via_LC(self, L0, C0, Ls_func): 
        return lambda phi: 1/np.sqrt((L0+Ls_func(phi))*C0)
    
    def generate_resonance_function_via_fit(self, p, f0, c2_func): 
        return lambda phi: 2*np.pi*f0/(np.sqrt(1+(p/(1-p))/c2_func(phi)))
        
    def generate_gsss_function(self, C0, p_func, res_func, c2_func, c3_func):
        '''
        source: https://arxiv.org/pdf/1806.06093.pdf
        (The frattini paper)
        calculates the g3 wrt flux given linear capacitance, participation ratio, and alpha
        return value is in Joules
        '''
        #calculate Ec
        Ec = self.e**2/(2*C0)
        return lambda phi: 1/6*p_func(phi)**2*c3_func(phi)/c2_func(phi)*np.sqrt(Ec*self.hbar*res_func(phi))
        
    def collect_TACO_data(self, gain_folder, plot = False, tla_pump = 0): 
        gain_cwd = gain_folder
        res = find_all_ddh5(gain_cwd)
        info_dict, bias_currents, best_gen_freqs, best_gen_powers, gains = superTACO_Bars(res, angles = [60,20], quanta_size = self.quanta_size, quanta_offset = self.quanta_offset, bardims = [0.001, 0.7], barbase = -24, plot = False)
        
        if plot: 
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(131)
            ax.plot(self.conv_func(bias_currents), np.array(best_gen_powers)-tla_pump, 'b.', markersize = 15)
            ax.set_title(r'Lowest 20dB Power (dBm) vs. Flux ($\Phi_0$)')
            ax.set_xlabel('Flux Quanta ($\Phi/\Phi_0)$')
            ax.set_ylabel('Generator Power @20dB Gain (dBm)')
            ax.grid()
        
        return bias_currents, best_gen_freqs, best_gen_powers-tla_pump, gains
        
    def g3_from_pump_power(self, 
                           dBgains: np.ndarray, 
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
        
        lin_pump_powers = np.power(10,pump_powers/10)*0.001 #pump power in watts
        #get the expected value of pump photons present in the resonator
        npTZC_arr = []
        numPumpPhotonsTZC = lin_pump_powers/(pump_omegas*self.hbar)*(np.sqrt(mode_kappas)/(mode_kappas/2-1j*(pump_detunings_from_res)))**2
        for val in numPumpPhotonsTZC:
            npTZC_arr.append(np.linalg.norm(val))
        numPumpPhotons = np.array(npTZC_arr)
        numPumpPhotonsDev = np.sqrt(8*mode_kappas*lin_pump_powers/(pump_omegas*self.hbar))/np.absolute(mode_kappas-2j*pump_detunings_from_res)
        Lin_Power_gains = np.power(10,dBgains/20)
        lpg = Lin_Power_gains
        
        g3_arr = -0.5*(mode_kappas/numPumpPhotons)*np.sqrt((np.sqrt(lpg)-1)/(np.sqrt(lpg)+1))
        
        
        return numPumpPhotonsDev, g3_arr, numPumpPhotons
    
    def process_ref_HFSS_sweep(self, HFSS_filepath, ref_port_name = 'B', lumped_port_name = 'sl', ind_name = 'Ls'): 
        data = pd.read_csv(HFSS_filepath)
        HFSS_dicts = []
        for inductance in np.unique(data[f'{ind_name} [pH]'].to_numpy()):
            filt = (data[f'{ind_name} [pH]'].to_numpy() == inductance)
            HFSS_dicts.append(dict(
                SNAIL_inductance = inductance,
                freq = data['Freq [GHz]'].to_numpy()[filt]*1e9,
                freqrad = data['Freq [GHz]'].to_numpy()[filt]*1e9*2*np.pi, #fitter takes rad*hz
                mag = data[f'mag(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt],
                phase = data[f'cang_deg_val(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt], 
                phaserad = data[f'cang_deg_val(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]*2*np.pi/360,
                dBmag = np.power(10, data[f'mag(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]/20),
                real = data[f'mag(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]*np.cos(data[f'cang_deg_val(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]*2*np.pi/360),
                imag = data[f'mag(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]*np.sin(data[f'cang_deg_val(S({ref_port_name},{ref_port_name})) []'].to_numpy()[filt]*2*np.pi/360),
                imY = data[f'im(Y({lumped_port_name},{lumped_port_name})) []'].to_numpy()[filt],
                ))
        return HFSS_dicts
        
    def fit_modes(self, *args, bounds = None, f0Guess_arr = None, Qguess = (1e2, 1e4), window_size = 600e6, plot = False):
        QextGuess, QintGuess = Qguess
        magBackGuess = 1
        
        HFSS_inductances, HFSS_res_freqs, HFSS_kappas = [], [], []
        
        for i, md in enumerate(args): 
            # print(type(f0Guess_arr))
            if type(f0Guess_arr) == np.ndarray: 
                f0Guess_arr = np.copy(f0Guess_arr)
                filt = (md['freqrad']>f0Guess_arr[i]-window_size/2)*(md['freqrad']<f0Guess_arr[i]+window_size/2)
                f0Guess = f0Guess_arr[i]
            else: 
                filt = np.ones(np.size(md['freqrad'])).astype(bool)
                # print(np.diff(md['phaserad']))
                # plt.plot(md['freq'][:-1]/1e9, np.diff(md['phaserad']))
                f0Guess = md['freq'][np.argmin(savgol_filter(np.gradient(md['phaserad']), 15,3))]*2*np.pi
                filt = (md['freqrad']>f0Guess-window_size/2)*(md['freqrad']<f0Guess+window_size/2)
                
            if bounds == None: 
                bounds = ([QextGuess / 10, QintGuess /10, f0Guess-500e6, magBackGuess / 2, 0],
                          [QextGuess * 10, QintGuess * 10, f0Guess+500e6, magBackGuess * 2, np.pi])
            
            popt, pcov = fit(md['freqrad'][filt], md['real'][filt], md['imag'][filt], md['mag'][filt], md['phaserad'][filt], Qguess = Qguess, f0Guess = f0Guess, phaseGuess = 0)
            if plot: 
                print("inductance: ", md['SNAIL_inductance'])
                plotRes(md['freqrad'][filt], md['real'][filt], md['imag'][filt], md['mag'][filt], md['phaserad'][filt], popt)
                
            Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
            kappa = popt[2]/2/np.pi/Qtot
            f0 = popt[2]/(2*np.pi)
            inductance = md['SNAIL_inductance']
            
            HFSS_inductances.append(inductance)
            HFSS_res_freqs.append(f0)
            HFSS_kappas.append(kappa)
            
            md['res_freq_rad'] = f0*2*np.pi
            md['kappa'] = kappa
            
        return HFSS_inductances, HFSS_res_freqs, HFSS_kappas
    
    def g3_from_admittance(self, Ej_large, c3_val, mds):
        phi_zpf_arr = []
        g3 = Ej_large*c3_val/6*(2*np.pi/self.phi0)**3
        for md in mds: 
            res_omega = md['res_freq_rad']
            # print("res_omega/2pi", res_omega/2/np.pi)
            omegas = md['freqrad']
            imY = md['imY']
            
            f_res_loc = np.argmin(np.abs(omegas-res_omega))
            slope = np.gradient(imY)[f_res_loc]/np.gradient(omegas)[f_res_loc]
            Zpeff = 2/(res_omega*slope)
    
    #         print("omega/2pi: ", res_omega/2/np.pi)
    #         print("slope: ", slope)
    #         print("Impedance: ", Zpeff)
            g3 *= np.sqrt(self.hbar/2*Zpeff)
            phi_zpf_arr.append(Zpeff)
        
        return g3
    
    def g3_from_admittance_raw(self, Ej_large, c3_val, res_omega):
        phi_zpf_arr = []
        g3 = Ej_large*c3_val/6*(2*np.pi/self.phi0)**3
        for res_omega in res_omegas: 
            res_omega = md['res_freq_rad']
            omegas = md['freqrad']
            imY = md['imY']
            
            f_res_loc = np.argmin(np.abs(omegas-res_omega))
            slope = np.gradient(imY)[f_res_loc]/np.gradient(omegas)[f_res_loc]
            Zpeff = 2/(res_omega*slope)
    
    #         print("omega/2pi: ", res_omega/2/np.pi)
    #         print("slope: ", slope)
    #         print("Impedance: ", Zpeff)
            g3 *= np.sqrt(self.hbar/2*Zpeff)
            phi_zpf_arr.append(Zpeff)
        
        return g3
        
    
if __name__ == '__main__': 
    SA = SnailAmp()
    HFSS_filepath = r'D:\HFSS_Sims\SA_2X\mode_s.csv'
    HFSS_dicts = SA.process_HFSS_sweep(HFSS_filepath)
    #fit all of them, try to choose a guess frequency and Q's that cooperate with all of them
    HFSS_inductances, HFSS_res_freqs, HFSS_kappas = SA.fit_modes(*HFSS_dicts, 
                                                                  Qguess = (5e1,1e3), 
                                                                  window_size = 100e6, 
                                                                  plot = True, 
                                                                  f0Guess_arr = None)
    HFSS_inductances = np.array(HFSS_inductances)
    HFSS_kappas = np.array(HFSS_kappas)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        