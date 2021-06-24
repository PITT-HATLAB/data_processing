# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:18:59 2021

@author: Ryan Kaufman - Hatlab
"""
from plottr.apps.autoplot import main

from data_processing.AWG_and_Alazar import Pulse_Processing_utils as PU
from measurement_modules.Helper_Functions import find_all_ddh5
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")

#%%

# datadir = r'E:\Data\Cooldown_20210611\SNAIL_Amps\C1\phase_preserving_checks\2021-06-17\amp_off\33_pt_sweep\2021-06-17'
datadir = r'E:\Data\Cooldown_20210611\SNAIL_Amps\C1\phase_preserving_checks\20dB\amp_on\33pt_sweep_with_switch_adjusted\2021-06-23'
filepaths = find_all_ddh5(datadir)

#%%
names = []
even_fits = []
odd_fits = []
even_contours = []
odd_contours = []
for filepath in filepaths: 
    bins_even, bins_odd, h_even, h_odd, guessParam = PU.extract_2pulse_histogram_from_filepath(filepath, hist_scale = 0.01, numRecords = 3840*2)
    even_fits.append(PU.fit_2D_Gaussian(bins_even, h_even, 
                                         guessParam[0], 
                                         max_fev = 1000))
    
    odd_fits.append(PU.fit_2D_Gaussian(bins_odd, h_odd, 
                                        guessParam[1],
                                        max_fev = 1000))
    names.append(filepath.split('rotation')[-1].split('.ddh5')[0])
#%%
fig = plt.figure(figsize = (18,12))
ax = fig.add_subplot(111)
ax.set_aspect(1)
for i, fit in enumerate(even_fits+odd_fits): 
    print(fit.info_dict['contour'][0])
    try: 
        ax.plot(fit.info_dict['contour'][0], fit.info_dict['contour'][1], label = names[i])
    except: 
        ax.plot(fit.info_dict['contour'][0], fit.info_dict['contour'][1], label = names[i//2])
    fit.plot_on_ax(ax, color = 'black')
    ax.legend()

#%%
fig, ax = plt.subplots()
for gfit in gaussian_fit_classes_arr:
    gfit.plot_on_ax(ax, color = 'black')
S_off = Gaussian_fits[0]
S_on = Gaussian_fits[1]
mag_gain1 = np.linalg.norm(S_on[0].center_vec())/np.linalg.norm(S_off[0].center_vec())
mag_gain2 = np.linalg.norm(S_on[1].center_vec())/np.linalg.norm(S_off[1].center_vec())

print("Power gain 1 (dB): ", 20*np.log10(mag_gain1))
print("Power gain 2 (dB): ", 20*np.log10(mag_gain2))
avg_sigma_on = np.average(np.average(S_on[0].info_dict['sigma_x'], S_on[0].info_dict['sigma_y']))
print("avg_sigma_on/avg_sigma_off: ", S_on[0].info_dict['sigma_x'])

