from data_processing.ddh5_Plotting.utility_modules.FS_utility_functions import fit_fluxsweep
import numpy as np

fit = fit_fluxsweep(r'Z:/Data/N25_L3_SP_2/fluxsweeps/2022-07-20_0001_FS_-30dBm_wider_VNA_VNA.ddh5', r'Z:\Data\N25_L3_SP_2\fluxsweeps\fits', 'N25_L3_SP_2',phaseName='vna_phase', powerName='vna_power', currentName='bias_current', freqName='vna_frequency')
#%%
fit.initial_fit(5.8e9, magBackGuess=0.01, QextGuess=20, adaptive_window=True, adapt_win_size=600e6)

#%%
semi = fit.semiauto_fit(fit.currents, fit.vna_freqs, fit.undriven_vna_power, fit.undriven_vna_phase, fit.initial_popt, adaptive_window=1, adapt_win_size=600e6, savedata = True, debug = False)

[bias_currents, res_freqs, Qints, Qexts, magBacks] = semi

