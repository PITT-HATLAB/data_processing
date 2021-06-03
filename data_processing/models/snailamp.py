# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:43:45 2021

@author: Hatlab_3
"""
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from data_processing.models.SNAIL_supporting_modules.Participation_and_Alpha_Fitter import slider_fit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from measurement_modules.Helper_Functions import find_all_ddh5
from data_processing.ddh5_Plotting.TACO_multiplot_b1 import superTACO_Bars

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
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 4))
    c4exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c4exp)

def c3_func_gen_vectorize(alpha_val): #can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 4)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 3))
    c3exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c3exp)

def c2_func_gen_vectorize(alpha_val): 
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 3)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 2))
    c2exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c2exp)


class SnailAmp(): 
    def __init__(self, junction_sizes: tuple, quanta_offset: float, quanta_size: float, Jc = 0.8): #uA/um^2
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
        self.phi0 = np.pi*self.hbar/self.e
        
        self.s_size, self.l_size = junction_sizes
        self.quanta_offset = quanta_offset
        self.quanta_size = quanta_size
        #function for converting bias currents to quanta fractions
        self.conv_func = lambda c: (c-quanta_offset)/quanta_size
        
        self.alpha_from_sizes = self.s_size/self.l_size
        self.I0s, self.I0l = Jc*self.s_size*1e-6, Jc*self.l_size*1e-6
        
        self.Lss, self.Lsl = self.Ic_to_Lj(self.I0s), self.Ic_to_Lj(self.I0l)
        self.Ejs, self.Ejl = self.Ic_to_Ej(self.I0s), self.Ic_to_Ej(self.I0l)
        
        self.Ls0 = parallel(self.Lss, self.Lsl)
        
        self.c2_func, self.c3_func, self.c4_func = self.generate_coefficient_functions(self.alpha_from_sizes, res = 100)
        
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
        
    

    def slider_participation_fitter(self, stored_fits_filepath: str, fluxsweep_filepath: str, ret_sliders = False): 
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
                                                                 self.alpha_arr)
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
        
    def generate_coefficient_functions(self, alpha_val, res = int(100), plot = False, show_coefficients = False):
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
        print("Calculating expansion coefficients")
        start_time = timer()
        
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c4_arr = c4_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c4_func = interp1d(phi_ext_arr, c4_arr, 'quadratic')
        
        
        #c3: 
        start_time = timer()
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c3_arr = c3_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c3_func = interp1d(phi_ext_arr, c3_arr, 'quadratic')
        
        
        #c2: 
        start_time = timer()
        phi_ext_arr = np.linspace(0,2*np.pi, res)
        c2_arr = c2_func_gen_vectorize(alpha_val)(phi_ext_arr)
        end_time = timer()
        print(f"Elapsed time: {np.round(end_time-start_time, 2)} seconds")
        c2_func = interp1d(phi_ext_arr, c2_arr, 'quadratic')
        
        if plot: 
            plt.plot(phi_ext_arr, self.c2_func(phi_ext_arr), label = "c2")
            plt.plot(phi_ext_arr, self.c3_func(phi_ext_arr), label = "c3")
            plt.plot(phi_ext_arr, self.c4_func(phi_ext_arr), label = 'c4')
            
            plt.legend()
        
        return c2_func, c3_func, c4_func
    
    def set_linear_inductance(self, L0): 
        self.L0 = L0
    def set_linear_capacitance(self, C0): 
        self.C0 = C0
        
    def generate_inductance_function(self, Ls0, c2_func):
        return lambda phi: Ls0/c2_func(phi)
    
    def generate_resonance_function_via_LC(self, L0, C0, Ls_func): 
        return lambda phi: 1/np.sqrt((L0+Ls_func(phi))*C0)
    
    def generate_resonance_function_via_fit(self, p, f0, c2_func): 
        return lambda phi: 2*np.pi*f0/(np.sqrt(1+(p/(1-p))/c2_func(phi)))
    
    def generate_p_func(self, )
        
    def generate_gsss_function(self, C0, p_func, res_func, c2_func, c3_func):
        '''
        source: https://arxiv.org/pdf/1806.06093.pdf
        (The frattini paper)
        calculates the g3 wrt flux given linear capacitance, participation ratio, and alpha
        return value is in Joules
        '''
        #calculate Ec
        Ec = self.e**2/(2*C0)
        return lambda phi: 1/6*p**2*c3_func(phi)/c2_func(phi)*np.sqrt(Ec*self.hbar*res_func(phi))
        
    def collect_TACO_data(self, gain_folder, plot = False): 
        gain_cwd = gain_folder
        res = find_all_ddh5(gain_cwd)
        info_dict, bias_currents, best_gen_freqs, best_gen_powers, gains = superTACO_Bars(res, angles = [60,20], quanta_size = self.quanta_size, quanta_offset = self.quanta_offset, bardims = [0.001, 0.7], barbase = -24, plot = False)
        
        if plot: 
            fig2 = plt.figure(2)
            ax = fig2.add_subplot(131)
            total_line_attenuation = 72
            ax.plot(self.conv_func(bias_currents), np.array(best_gen_powers)-total_line_attenuation, 'b.', markersize = 15)
            ax.set_title(r'Lowest 20dB Power (dBm) vs. Flux ($\Phi_0$)')
            ax.set_xlabel('Flux Quanta ($\Phi/\Phi_0)$')
            ax.set_ylabel('Generator Power @20dB Gain (dBm)')
            ax.grid()
        
        return bias_currents, best_gen_freqs, best_gen_powers, gains
        
    def g3_from_pump_power(self, 
                           gains: np.ndarray, 
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
        numPumpPhotons = np.sqrt(8*mode_kappas*lin_pump_powers/(pump_omegas*self.hbar))/np.absolute(mode_kappas-2j*pump_detunings_from_res)
        Lin_Power_gains = np.power(10,gains/20)
        lpg = Lin_Power_gains
        g3_arr = -0.5*(mode_kappas/numPumpPhotons)*np.sqrt((np.sqrt(lpg)-1)/(np.sqrt(lpg)+1))
        return numPumpPhotons, g3_arr
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        