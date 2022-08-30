# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:58:52 2022

@author: Hatlab-RRK
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect):
    omega0 = f0
    delta = freq - omega0
    S_11_up = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) - Qext / omega0
    S_11_down = 1.0 / (1j * delta * (2 + delta / omega0) / (1 + delta / omega0) + omega0 / Qint) + Qext / omega0
    S11 = magBack * (S_11_up / S_11_down) * np.exp(1j * (phaseCorrect))
    realPart = np.real(S11)
    imagPart = np.imag(S11)

    return (realPart + 1j * imagPart).view(np.float)

def reflectionFunc_re(freq, Qext, Qint, f0, magBack, phaseCorrect):
    return reflectionFunc(freq, Qext, Qint, f0, magBack, phaseCorrect)[::2]

def fit(freq, real, imag, mag, phase, Qguess=(2e4, 1e5),real_only = 0, bounds = None, f0Guess = None, magBackGuess = None, phaseGuess = np.pi, debug = False):
    # f0Guess = 2*np.pi*5.45e9
    # f0Guess = freq[np.argmin(mag)] #smart guess of "it's probably the lowest point"
    if f0Guess == None:
        f0Guess = freq[int(np.floor(np.size(freq)/2))] #dumb guess of "it's probably in the middle"
        # f0Guess = freq[np.argmin(mag)] #smart guess of "it's probably the lowest point"
    if debug: 
        print("Guess freq: "+str(f0Guess/(2*np.pi*1e9)))
    lin = 10**(mag / 20.0)
    if magBackGuess == None: 
        magBackGuess = np.average(lin[:int(len(freq) / 5)])
    # print(f"MAGBACKGUESS: {magBackGuess}")
    QextGuess = Qguess[0]
    QintGuess = Qguess[1]
    if bounds == None: 
        bounds=([QextGuess / 10, QintGuess /10, f0Guess/2, magBackGuess / 10.0, -2 * np.pi],
                [QextGuess * 10, QintGuess * 10, f0Guess*2, magBackGuess * 10.0, 2 * np.pi])
    
    target_func = reflectionFunc
    data_to_fit = (real  + 1j * imag).view(np.float)
    if real_only:
        target_func = reflectionFunc_re
        data_to_fit = real
    popt, pcov = curve_fit(target_func, freq, data_to_fit, 
                            p0=(QextGuess, QintGuess, f0Guess, magBackGuess, phaseGuess),
                            bounds=bounds,
                            maxfev=1e4, ftol=2.3e-16, xtol=2.3e-16)
    

    return popt, pcov


def plotRes(freq, real, imag, mag, phase, popt):
    xdata = freq / (2 * np.pi)
    realRes = reflectionFunc(freq, *popt)[::2]
    imagRes = reflectionFunc(freq, *popt)[1::2]
    # realRes = reflectionFunc(freq, *popt)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('real')
    plt.plot(xdata, real, '.')
    plt.plot(xdata, realRes)
    plt.subplot(1, 2, 2)
    plt.title('imag')
    plt.plot(xdata, imag, '.')
    plt.plot(xdata, imagRes)
    plt.show()

def process_ref_HFSS_sweep(HFSS_filepath, ref_port_name = 'B', lumped_port_name = 'sl', ind_name = 'Ls', ind_list = None, leakage = None): 
    '''
    for this to work you need an exported drivenmodal hfss sweep in csv format that has the following traces: 
        mag(S(ref,ref))
        cang_deg_val(S(ref,ref))
        im(Y(lumped, lumped))
        
    '''
    data = pd.read_csv(HFSS_filepath)
    HFSS_dicts = []
    if ind_list == None: 
        inductance_list = np.unique(data[f'{ind_name} [pH]'].to_numpy())
    else: 
        inductance_list = ind_list
        
    print("Inductance List")
        
    for inductance in inductance_list:
        if inductance == 1: 
            filt = (np.ones(np.size(data['Freq [GHz]'].to_numpy())).astype(bool))
        else: 
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
    
def fit_modes(*args, bounds = None, f0Guess_arr = None, Qguess = (1e2, 1e4), window_size = 100e6, plot = False):
    
    """
    takes in a mode dictionary from process_ref_HFSS_sweep above and fits each mode in a list
    
    for example of total code from start to finish for A mode of a SHARC:
        HFSS_filepath = r'G:\My Drive\amplifiers\Design Notebooks\SHARC_6X\6F1_mode_a_large.csv'
        HFSS_A_dicts = SA.process_ref_HFSS_sweep(HFSS_filepath, ref_port_name = 'U', lumped_port_name = 'sl', ind_name = 'Ls')
        A_inductances, A_res_freqs, A_kappas = SA.fit_modes(*HFSS_A_dicts, 
                                                                      Qguess = (1e3,1e4), 
                                                                      window_size = 400e6, 
                                                                      plot = True, 
                                                                      f0Guess_arr = None
                                                                    )
        A_inductances = np.array(A_inductances)
        A_kappas = np.array(A_kappas)
        
    """
    QextGuess, QintGuess = Qguess
    magBackGuess = 1
    
    HFSS_inductances, HFSS_res_freqs, HFSS_kappas = [], [], []
    
    for i, md in enumerate(args): 
        print(md.keys())
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
        plt.figure()           
        if bounds == None: 
            bounds = ([QextGuess / 10, QintGuess /10, f0Guess-500e6, magBackGuess / 2, 0],
                      [QextGuess * 10, QintGuess * 10, f0Guess+500e6, magBackGuess * 2, np.pi])
        
        popt, pcov = fit(md['freqrad'][filt], md['real'][filt], md['imag'][filt], md['mag'][filt], md['phaserad'][filt], Qguess = Qguess, f0Guess = f0Guess, phaseGuess = 0)
        if plot: 
            # print("inductance: ", md['SNAIL_inductance'])
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

if __name__ == '__main__': 
    HFSS_filepath = r'G:\My Drive\amplifiers\Design Notebooks\SNAIL_amps\SA_3X\SA_3X_C1_mode_s.csv'

    
    HFSS_dicts = process_ref_HFSS_sweep(HFSS_filepath, ref_port_name = 'B', lumped_port_name = 'sl', ind_name = 'Ls')
    guessfreqs = np.linspace(8.5e9, 7e9, len(HFSS_dicts))*2*np.pi
    A_inductances, A_res_freqs, A_kappas = fit_modes(*HFSS_dicts, 
                                                      Qguess = (1e2,1e4), 
                                                      window_size = 1000e6, 
                                                      plot = True, 
                                                      f0Guess_arr = None
                                                    )
    A_inductances = np.array(A_inductances)
    A_kappas = np.array(A_kappas)