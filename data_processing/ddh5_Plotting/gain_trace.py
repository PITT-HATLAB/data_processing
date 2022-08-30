# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:57:13 2021

@author: Hatlab-RRK
"""
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np



filepath = r'Z:/Data/SA_3C1_3221/Pulse_data_7GHz/gain_trace/2021-11-17/2021-11-17_0001_7GHz_gp1/2021-11-17_0001_7GHz_gp1.ddh5'

dd = all_datadicts_from_hdf5(filepath)['data']
pows = dd.extract('power')['power']['values']
freqs = dd.extract('power')['frequency']['values']
cf = 6.978e9
filt = (freqs>cf-20e6)*(freqs<cf+20e6)
bg = np.average(pows[freqs<6.925e9])

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(freqs[filt]/1e9, pows[filt]-bg)
ax.set_xlabel('VNA frequency (GHz)')

ax.set_ylabel('VNA Gain (dB)')
ax.legend()
ax.grid()
# ax.set_title(f'0.06mA, {8.65-30-10}dBm Cryo, +277kHZ generator detuning: 6.6MHz BW')

#%%
filepath = r'Z:/Data/N25_L3_SQ/traces/sat/2022-04-25/2022-04-25_0001_bp1_sat/2022-04-25_0001_bp1_sat.ddh5'

dd = all_datadicts_from_hdf5(filepath)['data']
pows = dd.extract('power')['power']['values']
freqs = dd.extract('power')['frequency']['values']

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(freqs-77, pows)
ax.set_xlabel('VNA power (dBm Cryo)')

ax.set_ylabel('VNA Gain (dB)')
ax.legend()
ax.grid()
ax.set_title('0.06mA, 8.65dBm RT, +277kHZ generator detuning: 6.6MHz BW')

#%% vna gen power sweep
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/vna_gain_sweep/2021-09-13/2021-09-13_0006_gain_vs_gen_power_50dBatten/2021-09-13_0006_gain_vs_gen_power_50dBatten.ddh5'

specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('power')['VNA_frequency']['values']
spec_powers = specData.extract('power')['power']['values']
gen_powers = specData.extract('power')['Gen_power']['values']
#take middle value for LO leakage
detuning = -1e6
center_freq = 6.096e9

lower_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-detuning)))]
IM_spur_lower = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-3*detuning)))]

upper_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+detuning)))]
IM_spur_upper = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+3*detuning)))]


lower_sideband_filt = spec_freqs == lower_sideband_freq
upper_sideband_filt = spec_freqs == upper_sideband_freq
IM_spur_lower_filt = spec_freqs == IM_spur_lower
IM_spur_upper_filt = spec_freqs == IM_spur_upper

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
for i, gp in enumerate(np.unique(gen_powers)): 
    gp_filt = gen_powers == gp
    center_freq = spec_freqs[np.argmax(spec_powers[gp_filt])]
    if i%5 ==0:
        ax.plot(spec_freqs[gp_filt], spec_powers[gp_filt], label = f'{gp} dBm')
highlight_freq = center_freq+detuning
ax.vlines(center_freq, 0, 30, linestyles = 'dashed', colors = 'black')
ax.vlines(highlight_freq, 0, 30, linestyles = 'dashed', colors = 'black')
# ax.plot(gen_powers[leakage_filt], spec_powers[leakage_filt], label = 'LO leakage (dBm)')
# ax.plot(gen_powers[upper_sideband_filt], spec_powers[upper_sideband_filt], label = 'Upper input tone power (dBm)')
# ax.plot(gen_powers[lower_sideband_filt], spec_powers[lower_sideband_filt], label = 'Lower input tone power (dBm)')
# ax.plot(gen_powers[IM_spur_upper_filt], spec_powers[IM_spur_upper_filt], label = 'Upper spur power (dBm)')
# ax.plot(gen_powers[IM_spur_lower_filt], spec_powers[IM_spur_lower_filt], label = 'Lower spur power (dBm)')

ax.set_xlabel('VNA Frequency (Hz)')
ax.legend()
ax.grid()
ax.set_ylabel('VNA Gain (dB)') 
ax.set_title(f'Amplifier Low-power gain: 20dB, f1-f2: {np.round(detuning/1e3)} kHz ')
ax.set_ylim([0,30])
ax.annotate(f'Signal input:\n{np.round((highlight_freq)/1e9, 5)} GHz', [highlight_freq, 20], [highlight_freq-2e6, 22], arrowprops=dict(facecolor='black', shrink=0.05))

#%% vna saturation gen power sweep
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/vna_saturation_sweep/2021-09-14/2021-09-14_0002_SH_5B1_saturation_sweep_offres_+500kHz/2021-09-14_0002_SH_5B1_saturation_sweep_offres_+500kHz.ddh5'


# sat_gen_freq = [gen_freq],
# sat_gen_power = [gen_power-gen_att],
# sat_vna_freq = [vna_cw_freq],
# sat_vna_powers = pows.reshape(1,-1)-vna_att,
# sat_gain = gains.reshape(1,-1),
# sat_phases = phases.reshape(1, -1)


specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('sat_gain')['sat_vna_powers']['values']
spec_powers = specData.extract('sat_gain')['sat_gain']['values']
gen_powers = specData.extract('sat_gain')['sat_gen_power']['values']
#take middle value for LO leakage
detuning = -0.5e6

# lower_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-detuning)))]
# IM_spur_lower = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-3*detuning)))]

# upper_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+detuning)))]
# IM_spur_upper = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+3*detuning)))]


# lower_sideband_filt = spec_freqs == lower_sideband_freq
# upper_sideband_filt = spec_freqs == upper_sideband_freq
# IM_spur_lower_filt = spec_freqs == IM_spur_lower
# IM_spur_upper_filt = spec_freqs == IM_spur_upper

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
for i, gp in enumerate(np.unique(gen_powers)): 
    print(i)
    gp_filt = gen_powers == gp
    print(gp_filt)
    # center_freq = spec_freqs[np.argmax(spec_powers[gp_filt])]
    print(spec_powers[gp_filt])
    ax.plot(spec_freqs[gp_filt][0], spec_powers[gp_filt][0], label = f'{gp+20} dBm')
        
# ax.vlines(center_freq, 0, 30, linestyles = 'dashed', colors = 'black')
# ax.vlines(center_freq-1e6, 0, 30, linestyles = 'dashed', colors = 'black')
# ax.plot(gen_powers[leakage_filt], spec_powers[leakage_filt], label = 'LO leakage (dBm)')
# ax.plot(gen_powers[upper_sideband_filt], spec_powers[upper_sideband_filt], label = 'Upper input tone power (dBm)')
# ax.plot(gen_powers[lower_sideband_filt], spec_powers[lower_sideband_filt], label = 'Lower input tone power (dBm)')
# ax.plot(gen_powers[IM_spur_upper_filt], spec_powers[IM_spur_upper_filt], label = 'Upper spur power (dBm)')
# ax.plot(gen_powers[IM_spur_lower_filt], spec_powers[IM_spur_lower_filt], label = 'Lower spur power (dBm)')

ax.set_xlabel('Input Signal Power (dBm RT)')
ax.legend()
ax.grid()
ax.set_ylabel('VNA S21 Power (dBm)') 
ax.set_title(f'Amplifier Low-power gain: 15dB, f1-f2: {np.round(detuning/1e3)} kHz ')
# ax.set_ylim([0,30])
# ax.annotate(f'Signal input:\n{np.round((center_freq-1e6)/1e9, 5)} GHz', [center_freq-1e6, 15], [center_freq-3e6, 17], arrowprops=dict(facecolor='black', shrink=0.05))
#%%same thing but with phase
 # vna saturation gen power sweep
filepath = r'Z:/Data/Hakan/SH_5B1_SS_Gain_6.064GHz/vna_saturation_sweep/2021-09-14/2021-09-14_0002_SH_5B1_saturation_sweep_offres_+500kHz/2021-09-14_0002_SH_5B1_saturation_sweep_offres_+500kHz.ddh5'


# sat_gen_freq = [gen_freq],
# sat_gen_power = [gen_power-gen_att],
# sat_vna_freq = [vna_cw_freq],
# sat_vna_powers = pows.reshape(1,-1)-vna_att,
# sat_gain = gains.reshape(1,-1),
# sat_phases = phases.reshape(1, -1)


specData = all_datadicts_from_hdf5(filepath)['data']
spec_freqs = specData.extract('sat_phases')['sat_vna_powers']['values']
spec_powers = specData.extract('sat_phases')['sat_phases']['values']
gen_powers = specData.extract('sat_phases')['sat_gen_power']['values']
#take middle value for LO leakage
detuning = -0.5e6

# lower_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-detuning)))]
# IM_spur_lower = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq-3*detuning)))]

# upper_sideband_freq = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+detuning)))]
# IM_spur_upper = np.unique(spec_freqs)[np.argmin(np.abs(np.unique(spec_freqs)-(center_freq+3*detuning)))]


# lower_sideband_filt = spec_freqs == lower_sideband_freq
# upper_sideband_filt = spec_freqs == upper_sideband_freq
# IM_spur_lower_filt = spec_freqs == IM_spur_lower
# IM_spur_upper_filt = spec_freqs == IM_spur_upper

#plot the LO leakage vs power
fig, ax = plt.subplots(figsize = (8,6))
for i, gp in enumerate(np.unique(gen_powers)): 
    print(i)
    gp_filt = gen_powers == gp
    print(gp_filt)
    # center_freq = spec_freqs[np.argmax(spec_powers[gp_filt])]
    print(spec_powers[gp_filt])
    ax.plot(spec_freqs[gp_filt][0], spec_powers[gp_filt][0], label = f'{gp+20} dBm')
        
# ax.vlines(center_freq, 0, 30, linestyles = 'dashed', colors = 'black')
# ax.vlines(center_freq-1e6, 0, 30, linestyles = 'dashed', colors = 'black')
# ax.plot(gen_powers[leakage_filt], spec_powers[leakage_filt], label = 'LO leakage (dBm)')
# ax.plot(gen_powers[upper_sideband_filt], spec_powers[upper_sideband_filt], label = 'Upper input tone power (dBm)')
# ax.plot(gen_powers[lower_sideband_filt], spec_powers[lower_sideband_filt], label = 'Lower input tone power (dBm)')
# ax.plot(gen_powers[IM_spur_upper_filt], spec_powers[IM_spur_upper_filt], label = 'Upper spur power (dBm)')
# ax.plot(gen_powers[IM_spur_lower_filt], spec_powers[IM_spur_lower_filt], label = 'Lower spur power (dBm)')

ax.set_xlabel('Input Signal Power (dBm RT)')
ax.legend()
ax.grid()
ax.set_ylabel('VNA S21 Power (dBm)') 
ax.set_title(f'Amplifier Low-power phase: 15dB Gain, f1-f2: {np.round(detuning/1e3)} kHz ')
# ax.set_ylim([0,30])
# ax.annotate(f'Signal input:\n{np.round((center_freq-1e6)/1e9, 5)} GHz', [center_freq-1e6, 15], [center_freq-3e6, 17], arrowprops=dict(facecolor='black', shrink=0.05))