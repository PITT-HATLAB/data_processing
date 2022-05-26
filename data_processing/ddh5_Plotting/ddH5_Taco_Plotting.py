# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:36:26 2020

@author: Hatlab_3
"""
# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as color
from scipy.ndimage import gaussian_filter

from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target, log_normalize_up_to_row
from data_processing.ddh5_Plotting.utility_modules.TACO_utility_functions import make_tacos, make_sat_img_plot, make_gain_profiles, make_gain_surface
import matplotlib.pyplot as plt
plt.rcParams.update({'font.weight': 'bold'})
plt.rc('axes', titlesize=15)  # fontsize of the axes titles
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

device_name = 'N25L3_SQ'

gain_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-02/2022-05-02_0001_0.0004mA_TACO_gain/2022-05-02_0001_0.0004mA_TACO_gain.ddh5'
gain_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0043_0.0004mA_TACO_16dB_gain.ddh5'
gain_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0027_0.0004mA_TACO_14dB_gain.ddh5'
# gain_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0011_0.0004mA_TACO_12dBm_gain.ddh5'
sat_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-02/2022-05-02_0002_0.0004mA_TACO_sat/2022-05-02_0002_0.0004mA_TACO_sat.ddh5'
sat_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0044_0.0004mA_TACO_16dB_sat.ddh5'
sat_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0038_0.0004mA_TACO_20dB_sat.ddh5'
# sat_filepath = r'Z:/Data/N25_L3_SQ/tacos/2022-05-03_0012_0.0004mA_TACO_12dBm_sat.ddh5'
#get files back out and into arrays
sat_dicts = all_datadicts_from_hdf5(sat_filepath)
satDict = sat_dicts['data']
gain_dicts = all_datadicts_from_hdf5(gain_filepath)
gainDict = gain_dicts['data']
# get saturation data
sat_data = satDict.extract('sat_gain')
[sat_bias_current, sat_gen_freq, sat_gen_power, sat_vna_powers, sat_gain, sat_vna_freq] = [sat_data.data_vals('sat_bias_current'), 
                                                                sat_data.data_vals('sat_gen_freq'),
                                                                sat_data.data_vals('sat_gen_power'), 
                                                                sat_data.data_vals('sat_vna_powers'), 
                                                                sat_data.data_vals('sat_gain'), 
                                                                sat_data.data_vals('sat_vna_freq')
                                                                ]
#Get Gain data
gain_data = gainDict.extract('calculated_gain')
[bias_current, gen_frequency, gen_power, calc_gain] = [gain_data.data_vals('bias_current'),
                                            gain_data.data_vals('gen_frequency'), 
                                            gain_data.data_vals('gen_power'), 
                                            gain_data.data_vals('calculated_gain')
                                            ]
#%%Extract slices of currents, saturation data, etc which are each individual tacos
b1_val = np.unique(bias_current)[0]
b1 = (bias_current == b1_val)
line_att = 77-30
# *(gen_power<11)*(gen_power>8.5)
""
target = 16
plt.style.use('hatlab')
gf1, gp1, g1 = gen_frequency[b1]/1000, gen_power[b1]-line_att, calc_gain[b1]
fig, ax, cb = make_tacos(b1_val, gf1, gp1, g1, vmin = target - 2, vmax = target+2, target = target)
ax.set_xlabel("Generator Frequency (GHz)")
ax.set_ylabel("Generator Pump Power (dBm)")
cb.set_label("Gain (dB)")
fancy = True
if fancy:
    title = f'{device_name} {target}dB Generator Power vs. Generator Frequency\nBias = {np.round(b1_val*1000, 4)}mA'
else:
    title = f'{get_name_from_path(gain_filepath)}\nSHARC41: Bias = {np.round(b1_val*1000, 4)}mA'
ax.title.set_text(title)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(linestyle = '--', zorder = 2)
#saturation plots 
bp1 = (sat_bias_current == np.unique(sat_bias_current)[0])
# plt.plot(bp1)
sf1, svp1, sgp1, sg1 = sat_gen_freq[bp1], sat_vna_powers[bp1], sat_gen_power[bp1], sat_gain[bp1]
fig, ax, img = make_sat_img_plot(b1_val, sf1/1000, svp1-line_att, sg1, norm_power = -65-line_att, levels = [-20, -1,1, 20], filter_window = 0, vmin = -1.7, vmax = 1.7)
#supplementary graph info
if fancy:
    title = f'{device_name} Saturation power vs. Generator Frequency\nBias = {np.round(b1_val*1000, 4)}mA'
else: 
    title = f'{get_name_from_path(sat_filepath)}\nBias = {np.round(b1_val*1000, 4)}mA'
cb = fig.colorbar(img, ax = ax)
ax.title.set_text(title)
plt.xlabel('Generator Frequency(GHz)')
plt.ylabel('Signal Power (dBm)')
cb.set_label(f"S21 change from {target}dB (dB)")

#%%plot a 3d profile of all the gain traces that were the "best"
# make_gain_profiles(gain_filepath, angles = [20, 45])

fig, ax = make_gain_surface(gain_filepath)
ax.azim = 90
ax.elev = 0
#%%Plot individual power sweeps to check
gain_traces = satDict.extract('sat_gain').data_vals('sat_gain')
vna_freqs = satDict.extract('sat_gain').data_vals('sat_vna_freq')
gen_frequency = satDict.extract('sat_gain').data_vals('sat_gen_freq')
sat_currents = satDict.extract("sat_gain").data_vals('sat_bias_current')
b1 = np.isclose(sat_currents, -0.02e-3, 
colors = [color.hex2color('#0000FF'), color.hex2color('#FFFFFF'), color.hex2color('#FF0000')]
_cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
f_val = 14.815e9
f1 = gen_frequency == gen_frequency[np.argmin(np.abs(gen_frequency-f_val))]
vnaf1, gp2, g2 = vna_freqs[b1*f1]/1e6, gen_power[b1*f1], gain_traces[b1*f1]-20
# plt.pcolormesh(vnaf1, gp2, g2, cmap = _cmap, vmin = -3, vmax = 3)
index = 3
plt.plot(vnaf1, g2)
plt.colorbar()
plt.xlabel('VNA Frequency (MHz)')
plt.ylabel('Gen Power (dBm)')
plt.title(f'Power sweep at bias = {np.round(b1_val*1000, 3)}mA, Gen Frequency {f_val/1e6}MHz')

#%%Plot Individual VNA Traces
gain_data = gainDict.extract('calculated_gain')
gain_traces = gainDict.extract('gain_trace').data_vals('gain_trace')
gen_power = gainDict.extract('gain_trace').data_vals('gen_power')
vna_freqs = gainDict.extract('gain_trace').data_vals('vna_frequency')
gen_frequency = gainDict.extract('gain_trace').data_vals('gen_frequency')
gp_val = -29+line_att
gp_filt = np.isclose(gen_power, gp_val, atol = 0.05)
f_val= np.round(np.average(gen_frequency[gp_filt])/1e6, 0)
plt.plot(vna_freqs[gp_filt][0]/1e9, np.convolve(gain_traces[gp_filt][0], np.blackman(1)))
plt.xlabel("VNA frequency (GHz)")
plt.ylabel("Gain (dB)")
plt.grid()
# plt.plot(vna_freqs[gp_filt][0]/1e6, gain_traces[gp_filt][0])
plt.title(f'Trace at bias = {np.round(b1_val*1000, 3)}mA\nGen Frequency {f_val}MHz\nGen Power {gp_val}dBm')
#%%Plot Individual Saturation traces
