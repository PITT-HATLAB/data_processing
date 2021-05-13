# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:43:45 2021

@author: Hatlab_3
"""
import numpy as np 
import matplotlib.pyplot as plt
# import sympy as sp

def parallel(v1, v2): 
    return 1/(1/v1+1/v2)
    
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
        
    def Ic_to_Ej(self, Ic):
        return Ic*self.phi0/(2*np.pi)
    
    def Ic_to_Lj(self, Ic): 
        return self.phi0/(2*np.pi*Ic)
        
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

