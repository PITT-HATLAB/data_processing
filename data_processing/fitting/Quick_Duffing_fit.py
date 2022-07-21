from data_processing.ddh5_Plotting.ddH5_Duff_Plotting_v2 import fit_Duff_Measurement
import numpy as np
import matplotlib.pyplot as plt

FS_filepath = r'Z:/Data/N25_L3_SP_2/fluxsweeps/fits/2022-07-20_0016_N25_L3_SP_2.ddh5'
Duff_filepath = 'Z:/Data/N25_L3_SP_2/duffing/2022-07-21_0001_N25_L3_SP_2_Duff_VNA_VNA.ddh5'
save_dir = r'Z:\Data\N25_L3_SP_2\duffing\fits'

DFit = fit_Duff_Measurement('N25_L3_SP_2_Duff_fit')

DFit.load_data(Duff_filepath, FS_filepath, current_filt = [-10e-3,10e-3], 
               current_name = 'amp_bias', 
               gen_power_name = 'gen_power',
               vna_freq_name = 'vna_frequency',
               vna_pow_name = 'vna_power',
               vna_phase_name = 'vna_phase'
               )
#%%
DFit.initial_fit(5.8e9, 
                  QextGuess = 30, 
                  QintGuess = 2000, 
                  magBackGuess = 0.001, 
                  bounds = None, 
                  smooth = False, 
                  smooth_win = 11,
                  phaseOffGuess = np.pi, 
                  debug = True, 
                  adaptive_window = True, 
                  adapt_win_size = 400e6
                )

#%%
DFit.create_file(save_dir)
DFit.fit(
        debug = False, 
        save_data = True,
        max_gen_power = 15, 
        smooth = False, 
        smooth_win = 101, 
        adaptive_window = False,  
        adapt_win_size = 800e6,  
        fourier_filter = False, 
        bounds = None, 
        fourier_cutoff = 40, 
        pconv_tol = 300, 
        accept_low_conv = True)

#%%
#pull information out of the fit file and plot it
from plottr.data.datadict_storage import all_datadicts_from_hdf5
save_filepath = r'Z:/Data/N25_L3_SP_2/duffing/fits/2022-07-21_0009_N25_L3_SP_2_Duff_fit.ddh5'
dd = all_datadicts_from_hdf5(save_filepath)['data']

res_shift_ref_low = dd.extract('res_shift_ref_low').data_vals('res_shift_ref_low')
res_shift_ref_undriven = dd.extract('res_shift_ref_undriven').data_vals('res_shift_ref_undriven')
current = dd.extract('res_shift_ref_low').data_vals('current')
gen_power = dd.extract('res_shift_ref_low').data_vals('gen_power')

uc = np.unique(current)
up = np.unique(gen_power)

ucs = np.size(np.unique(current))
ups = np.size(np.unique(gen_power))
scale = 150
x, y = np.meshgrid(uc, up)
fig, ax = plt.subplots()
img = ax.pcolormesh(x*1000, y, res_shift_ref_low.reshape((ups, ucs))/1e6, vmin = -scale, vmax = scale, cmap = 'seismic')
# ax.set_xlabel(r'Flux Bias ($\frac{\Phi}{\Phi0}$)')
ax.set_label('Flux Bias (mA)')
ax.set_ylabel('Generator  Power (dBm RT)')
cb = plt.colorbar(img, ax = ax)
cb.set_label('Resonant Frequency Shift (MHz)')
ax.set_title('Duffing Test')

fig, ax = plt.subplots()
ax.pcolormesh(x*1000, y, res_shift_ref_undriven.reshape((ups, ucs))/1e6, vmin = -scale, vmax = scale, cmap = 'seismic')
# ax.set_xlabel(r'Flux Bias ($\frac{\Phi}{\Phi0}$)')
ax.set_label('Flux Bias (mA)')
ax.set_ylabel('Generator  Power (dBm RT)')
cb = plt.colorbar(img, ax = ax)
cb.set_label('Resonant Frequency Shift (MHz)')
ax.set_title('Duffing Test')