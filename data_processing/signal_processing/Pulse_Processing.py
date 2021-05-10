# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:17:43 2021

@author: Hatlab_3
"""
import numpy as np
import time

import matplotlib.pyplot as plt
    
def demod(signal_data, reference_data, mod_freq = 50e6, sampling_rate = 1e9): 
    '''
    #TODO: 
    Parameters
    ----------
    signal_data : np 1D array - float64
        signal datapoints
    reference_data : np 1D array - float64
        reference datapoints
    mod_freq : float64, optional
        Modulation frequency in Hz. The default is 50e6.
    sampling_rate : float64, optional
        sampling rate in samples per second. The default is 1e9.

    Returns
    -------
    sig_I_summed : np 1D array - float64
        Signal multiplied by Sine and integrated over each period.
    sig_Q_summed : np 1D array - float64
        Signal multiplied by Cosine and integrated over each period.
    ref_I_summed : np 1D array - float64
        reference multiplied by sine and integrated over each period.
    ref_Q_summed : np 1D array - float64
        Reference multiplied by Cosine and integrated over each period.
    '''
    
    #first demodulate both channels
    point_number = np.arange(np.size(signal_data))
    period = int(sampling_rate/mod_freq)
    # print('Modulation period: ', period)
    SinArray = np.sin(2*np.pi/period*point_number)
    CosArray = np.cos(2*np.pi/period*point_number)
    
    sig_I = signal_data*SinArray
    sig_Q = signal_data*CosArray
    ref_I = reference_data*SinArray
    ref_Q = reference_data*CosArray
    
    #now you cut the array up into periods of the sin and cosine modulation, then sum within one period
    #the sqrt 2 is the RMS value of sin and cosine squared, period is to get rid of units of time
    
    sig_I_summed = np.sum(sig_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    sig_Q_summed = np.sum(sig_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_I_summed = np.sum(ref_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_Q_summed = np.sum(ref_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    
    return (sig_I_summed, sig_Q_summed, ref_I_summed, ref_Q_summed)

def phase_correction(sigI, sigQ, refI, refQ): 
    '''
    Parameters
    ----------
    sigI : np 2D array - (records,samples) float64
        demodulated signal - In-phase
    sigQ : np 2D array - (records,samples) float64
        demodulated signal - Quadrature phase
    refI : np 2D array - (records,samples) float64
        demodulated reference - In-phase
    refQ : np 2D array - (records,samples) float64
        demodulated reference - quadrature-phase
        
    Note: reference and signal arrays must all be of the same length

    Returns
    -------
    sigI_corrected : np 2D array - float64
        Signal I rotated by reference phase averaged over each record
    sigQ_corrected : np 2D array - float64
        Signal Q rotated by reference phase averaged over each record

    '''
    sigI_corrected = np.zeros(np.shape(sigI))
    sigQ_corrected = np.zeros(np.shape(sigQ))
    rI_trace = np.zeros(np.shape(sigI)[0])
    rQ_trace =np.zeros(np.shape(sigI)[0])
    for i, (sI_rec, sQ_rec, rI_rec, rQ_rec) in enumerate(zip(sigI,sigQ,refI,refQ)):
        
        rI_avg, rQ_avg = np.average(rI_rec), np.average(rQ_rec)
        
        rI_trace[i], rQ_trace[i] = rI_avg, rQ_avg
        
        Ref_mag = np.sum(np.sqrt(rI_avg**2 + rQ_avg**2))
        
        sigI_corrected[i] = (sI_rec*rI_avg + sQ_rec*rQ_avg)/Ref_mag
        sigQ_corrected[i] = (-sI_rec*rQ_avg + sQ_rec*rI_avg)/Ref_mag
    
    return sigI_corrected, sigQ_corrected, rI_trace, rQ_trace
