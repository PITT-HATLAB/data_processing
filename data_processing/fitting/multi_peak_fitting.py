# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:46:40 2022

@author: Hatlab-RRK

write code that can process a pump scan to find multiple peaks in gain traces

"""
import numpy as np
from scipy.signal import find_peaks
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
sweep_fp = r'Z:/Data/N25_L3_SP_3/pump_scans/2022-06-17_0003_N25_L3_SP_3_Pump_both_VNA.ddh5'
dd = all_datadicts_from_hdf5(sweep_fp)['data']

vna_freqs = dd.extract('vna_power')['vna_frequency']['values']
gen_pows = dd.extract('vna_power')['gen_power']['values']
gen_det = dd.extract('vna_power')['Generator_detuning']['values']
amp_bias = dd.extract('vna_power')['amp_bias']['values']

vna_power = dd.extract('vna_power')['vna_power']['values']
#%%
amp_bias_val = 0.7777e-3
gen_det_val = 80e6
gen_pow_val = 11
gen_pow_norm_val = -5
filt = (amp_bias == amp_bias_val)*(gen_det == gen_det_val)*(gen_pows == gen_pow_val)
norm_filt = (amp_bias == amp_bias_val)*(gen_det == gen_det_val)*(gen_pows == gen_pow_norm_val)

plt_pwrs = vna_power[filt]-vna_power[norm_filt]

plt.plot(vna_freqs[filt], vna_power[filt])
#%% find peaks
peak_inds, peaks_dict = find_peaks(plt_pwrs, prominence = 3, height = 10)
plt.plot(vna_freqs[filt], plt_pwrs)
plt.plot(vna_freqs[filt][peak_inds], plt_pwrs[peak_inds], '.')

#%%fit the whole trace with the assistance of the find_peaks input

from scipy.optimize import curve_fit
gain_func = lambda x, G, f, bw: G/(1+((x-f)/bw)**2)

def mp_gain_func(par_arr_1d, freqs, n_peaks = 1):
    size = np.size(par_arr_1d)
    x_arr, G_arr, f_arr, bw_arr = par_arr_1d.reshape(n_peaks, size//n_peaks)
    trace = np.empty(np.size(x_arr[0]))
    for x, G, f, bw in zip(x_arr, G_arr, f_arr, bw_arr): 
        trace+=gain_func(freqs, G, f, bw)
    return trace

def create_mp_gain_func(n_peaks): 
    return lambda par_arr_1d, mp_gain_func()

def fit_lor_from_peaks(peak_inds, peak_dicts, powers, freqs, Qguess = 100): 
    guess = []
        

#%%
freqs = self.getSweepData()



popt, pcov = curve_fit(gain_func, freqs, 10**(mag/10), p0 = [100, np.average(freqs), np.average(freqs)/10])
G, f, bw = popt
if plot: 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot((freqs-f)/1e6, mag, label = 'data')
    ax.plot((freqs-f)/1e6, 10*np.log10(gain_func(freqs, *popt)), label = 'Lorentzian Fit')
    ax.set_xlabel("Frequency Detuning (MHz)")
    ax.set_ylabel("Gain (dB)")
    ax.legend()
    ax.grid(b = 1)

return 10*np.log10(G), f/1e9, np.abs(2*bw/1e6), popt, gain_func