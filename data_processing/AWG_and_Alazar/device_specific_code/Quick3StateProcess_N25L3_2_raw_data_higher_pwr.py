# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:11:55 2022

@author: Hatlab-RRK

Purpose: create a neat, almost-executable file that can quickly plot a 3-state pulse file, and have the option to do just histograms, 
or additionally try to fit using majority vote and give classification accuracy
"""
import data_processing.signal_processing.Pulse_Processing_utils_raw_data as pulseUtils
from data_processing.signal_processing.Pulse_Processing import demod_period, demod_all_records, phase_correction, filter_all_records, remove_offset, generate_matched_weight_funcs, weighted_histogram, fit_2D_Gaussian, Gaussian_2D, hist_discriminant, majorityVote3State, bpf_func
from matplotlib.patches import Ellipse
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, sosfilt, sosfreqz
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#%%

#power sweep on the wideband mode
f = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/multi-rep_s_lin/2022-06-09_0074_pwr_sweep_Sig_Volt_0.08_V_Rep_1__.ddh5'
f = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb/power_sweep_+30MHz/2022-05-27_0002_pwr_swp_0dB_att_mod-50_Sig_Volt_0.3_V_.ddh5'
# f = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/multi-rep_s_lin/2022-06-09_0092_pwr_sweep_Sig_Volt_0.09_V_Rep_9__.ddh5'
f = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/multi-rep_s_and_i/0.032/combined_demod/2022-06-08_0001_demod_0.032V.ddh5'
time, signal_arr, ref_arr = pulseUtils.Process_One_Acquisition_3_state(f)
#%% see if there are multiple freqs in the data
sGAvg = np.average(signal_arr[0], axis = 0)
fwindow = [0, 200]
freqs = fftfreq(4096, 1e-9)/1e6
ffilt = (freqs<fwindow[1])*(freqs>fwindow[0])
plt.figure()
plt.plot(freqs[ffilt], 20*np.log10(np.abs(fft(sGAvg))/np.sqrt(50))[ffilt])
plt.title("Spectrum before filtering")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBm)')
plt.ylim(-60, 40)
plt.vlines(50, -20, 20)

#%% generate a filter to select a frequency spectrum area of the data
cf, BW, order = 50e6, 10e6, 1
filt = bpf_func(cf, BW, order) #center frequency, width, order of butterworth filter
w, h = sosfreqz(filt, fs = 1e9)
ffilt2 = (w/1e6>fwindow[0])*(w/1e6<fwindow[1])
plt.figure()
plt.plot(w[ffilt2]/1e6, 10*np.log10(np.abs(h)**2)[ffilt2])
plt.title(f"Applied Butterworth filter {cf/1e6}MHz, {BW/1e6}MHz wide, order {order}")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude $|H|^2$ (dB')
plt.ylim(-60, 5)
plt.figure()
plt.plot(w[ffilt2]/1e6,np.unwrap(np.angle(h))[ffilt2])
plt.title(f"Applied Butterworth filter {cf/1e6}MHz, {BW/1e6}MHz wide, order {order}")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude $<H$ (dB')
#%% apply the filter, demodulate, remove offsets, and plot averages
apply_filter = 0
signal_arr_filtered = []
ref_arr_filtered = []
state_data_arr = []
state_ref_arr = []
for sdata, rdata in zip(signal_arr, ref_arr): 
    #bc I got rid of the 50MHz bandpass theres a dc offset:
    sdata_dc_off = [s_trace-np.average(s_trace[0:100]) for s_trace in sdata]
    rdata_dc_off = [r_trace-np.average(r_trace[0:100]) for r_trace in rdata]
    
    if apply_filter: 
        sdata_filt, rdata_filt = filter_all_records(sdata_dc_off, rdata_dc_off, filt) #bw is 10MHz
    else: 
        sdata_filt, rdata_filt = sdata_dc_off, rdata_dc_off
    signal_arr_filtered.append(sdata_filt)
    ref_arr_filtered.append(rdata_filt)
    demod_data, demod_ref = demod_all_records(sdata_filt, rdata_filt, period = 20, sig_demod_freq = cf, ref_demod_freq = cf)
    state_data_arr.append(demod_data)
    state_ref_arr.append(demod_ref)
    
G_data, E_data, F_data = state_data_arr
G_ref, E_ref, F_ref = state_ref_arr
# phase correction
pc = 1
if pc: 
    G_I_corr, G_Q_corr, rI_trace, rQ_trace = phase_correction(*G_data, *G_ref, debug = True)
    E_I_corr, E_Q_corr, rI_trace, rQ_trace = phase_correction(*E_data, *E_ref)
    F_I_corr, F_Q_corr, rI_trace, rQ_trace = phase_correction(*F_data, *F_ref)
else: 
    [G_I_corr, G_Q_corr], [E_I_corr, E_Q_corr], [F_I_corr, F_Q_corr] = G_data, E_data, F_data 

G_data_off, E_data_off, F_data_off = np.array([G_I_corr, G_Q_corr]), np.array([E_I_corr, E_Q_corr]), np.array([F_I_corr, F_Q_corr])
#remove offsets then average
G_data, E_data, F_data = [remove_offset(*data, window = [0, 50]) for data in [G_data_off, E_data_off, F_data_off]]
GAvg, EAvg, FAvg = [np.average(data, axis = 1) for data in [G_data, E_data, F_data]]

fig, axs = plt.subplots(2)
axs[0].plot(GAvg[0], label = 'G')
axs[0].plot(EAvg[0], label = 'E')
axs[0].plot(FAvg[0], label = 'F')
axs[0].set_title("I")

axs[1].plot(GAvg[1], label = 'G')
axs[1].plot(EAvg[1], label = 'E')
axs[1].plot(FAvg[1], label = 'F')
axs[1].set_title("Q")
fig.tight_layout()
fig, ax = plt.subplots()

ax.plot(GAvg[0], GAvg[1], label = 'G')
ax.plot(EAvg[0], EAvg[1], label = 'E')
ax.plot(FAvg[0], FAvg[1], label = 'F')
ax.legend(bbox_to_anchor = (1,1))
ax.set_aspect(1)
is_filt = {0:"off", 1: 'on'}
ax.set_title(f"Average of data with filter {is_filt[apply_filter]}")

#%%check the effect of the filter
sGAvg2 = np.average(signal_arr_filtered[0], axis = 0)
plt.figure()
plt.plot(freqs[ffilt], 20*np.log10(np.abs(fft(sGAvg2)[ffilt])))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBm)')
plt.title("Spectrum after filtering")

#%% This marks the stage where the data used to be saved (without a digital filter), all the rest is linear processing and majority vote


#%% histogram the data with weight functions
GE, GF, EF = [generate_matched_weight_funcs(*data, bc = False) for data in [[G_data, E_data], [G_data, F_data], [E_data, F_data]]]
alldata = np.append(np.append(G_data, E_data, axis = 1), F_data, axis = 1)
# fig2, axs2 = plt.subplots(3, 3, figsize = (6,12))
titles = np.array([['SGE_'+state, 'SGF_'+state, 'SEF_'+state] for state in ['G', 'E', 'F']]).T.flatten()
titles_all = np.array(['SGE_all', 'SGF_all', 'SEF_all'])
wfs = np.repeat([GE,GF,EF], 3, axis = 0)
avgs = np.tile([GAvg,EAvg,FAvg], (3, 1, 1))
scale = 0.003
nb = 300

h_arr = [
    weighted_histogram(WF[0], WF[1], data[0], data[1], scale = scale, plot = 0, num_bins = nb) for 
    WF, data in 
    zip(
    np.repeat([GE,GF,EF], 3, axis = 0),
    np.tile([G_data, E_data, F_data], (3,1,1,1))
    )]

h_arr_all =[
    weighted_histogram(WF[0], WF[1], data[0], data[1], scale = scale, plot = 0, num_bins = nb) for 
    WF, data in 
    zip(
    [GE,GF,EF],
    [alldata, alldata, alldata]
    )]
fig, axs = plt.subplots(3, 3, figsize = (12,12))
axs = axs.T.flatten()
for i, h in enumerate(h_arr): 
    #plot the histograms
    ax = axs[i]
    [bins, h_2d, Ipts, Qpts] = h
    if i ==0 : 
        bins_universal = bins #need to reference this later
    im = ax.pcolormesh(bins, bins, h_2d)
    divider = make_axes_locatable(ax)
    ax.set_aspect(1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax.set_title(titles[i])
    A_y0Guess = np.dot(avgs[i][0], wfs[i][0])+np.dot(avgs[i][1], wfs[i][1])
    A_x0Guess = np.dot(avgs[i][1], wfs[i][0])-np.dot(avgs[i][0], wfs[i][1])
    ax.arrow(0, 0, A_x0Guess, A_y0Guess, length_includes_head = True, width = scale/20, head_width = scale/20)
fig.tight_layout()
#%%fitting
gaussians = []
gaussian_hist = []
wfs = np.repeat([GE,GF,EF], 3, axis = 0)
avgs = np.tile([GAvg,EAvg,FAvg], (3, 1, 1))
max_fev = 10000
fig, axs = plt.subplots(3, 3, figsize = (12,12))
axs = axs.T.flatten()
for i, h in enumerate(h_arr): 
    #plot the histograms
    ax = axs[i]
    [bins, h_2d, Ipts, Qpts] = h
    if i ==0 : 
        bins_universal = bins #need to reference this later
    im = ax.pcolormesh(bins, bins, h_2d)
    divider = make_axes_locatable(ax)
    ax.set_aspect(1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax.set_title(titles[i])
    

    
    A_y0Guess = np.dot(avgs[i][0], wfs[i][0])+np.dot(avgs[i][1], wfs[i][1])
    A_x0Guess = np.dot(avgs[i][1], wfs[i][0])-np.dot(avgs[i][0], wfs[i][1])
    
    
    A_ampGuess = np.average(np.sqrt(np.array(Ipts)**2+np.array(Qpts)**2))
    A_sxGuess = np.sqrt(np.var(Ipts)+np.var(Qpts))/4
    # A_thetaGuess = np.average(np.angle(A_x0Guess+1j*A_y0Guess))
    A_thetaGuess = 0 
    
    print("\n\nx0 Guess: ", A_x0Guess)
    print("y0 Guess: ", A_y0Guess)
    print("sigma Guess: ", A_sxGuess)
    print("amplitude guess ", A_ampGuess)
    print("\n\n")
    
    guessParams = [A_ampGuess, A_x0Guess, A_y0Guess, A_sxGuess]
    #do a 2D Gaussian Fit to the data
    gaussians.append(fit_2D_Gaussian(titles[i], bins, h_2d, 
                                                    guessParams,
                                                    # None,
                                                    max_fev = max_fev,
                                                    contour_line = 2, debug = 1) 
                                                    )
    print('popt', gaussians[-1].info_dict['popt'])
    gaussian_hist.append(Gaussian_2D(np.meshgrid(bins[:-1], bins[:-1]), *gaussians[-1].info_dict['popt']))
    
    print('popt: ', gaussians[-1].print_info())

#%% plot the discriminants
for i in range(np.shape(gaussian_hist)[0]): 
    print(i, gaussians[i].info_dict['name'])
#pair them up by name, format S{data1}{data2}_{data_classified}
gaussian_pairs = [[[gaussians[i], gaussians[j]], 
                   [gaussian_hist[i], gaussian_hist[j]], 
                   [h_arr[i][1], h_arr[j][1]]] for i, j in [(0,1), (3, 5), (7,8)]]



fig, axs = plt.subplots(2, 3, figsize = (9, 4))
viridis = cm.get_cmap('magma', 256)
newcolors = viridis(np.linspace(0, 1, 256))
gray = np.array([0.1, 0.1, 0.1, 0.1])
newcolors[128-5: 128+5] = gray
newcmp = ListedColormap(newcolors)

vote_maps = []

for i, [fits, gen_hist, data_hist] in enumerate(gaussian_pairs): 
    [h1_fit, h2_fit] = fits
    [h1_gen, h2_gen] = gen_hist
    [h1_data, h2_data] = data_hist 
    #plot the generated
    print(np.shape(h2_data))
    pc0 = axs[0, i].pcolormesh(bins_universal, bins_universal, h1_data+h2_data)
    
    scale = np.max((h1_gen, h2_gen))
    hist_disc = hist_discriminant(h1_gen, h2_gen)
    vote_maps.append(hist_disc)
    
    pc1 = axs[1, i].pcolormesh(bins_universal, bins_universal, (h1_gen+h2_gen)*(hist_disc-1/2)/scale*5, cmap = newcmp, vmin = -1, vmax = 1)
    plt.colorbar(pc0, ax = axs[0, i],fraction=0.046, pad=0.04)
    plt.colorbar(pc1, ax = axs[1, i],fraction=0.046, pad=0.04)
    h1_fit.plot_on_ax(axs[1, i])
    axs[1, i].add_patch(h1_fit.sigma_contour())
    h2_fit.plot_on_ax(axs[1,i])
    axs[1, i].add_patch(h2_fit.sigma_contour())
    axs[0, i].set_aspect(1)
    axs[1, i].set_aspect(1)
    
fig.tight_layout()

print("")

#%% Majority vote
#G_data, E_data, F_data = [G_I_corr, G_Q_corr], [E_I_corr, E_Q_corr], [F_I_corr, F_Q_corr]
states = [G_data, E_data, F_data]
alldata_I, alldata_Q = [np.append(np.append(states[0][i], states[1][i], axis = 0), states[2][i], axis = 0) for i in [0, 1]]#shape of [N_records, N_time_points]
fid_G, fid_E, fid_F, numberNull = majorityVote3State(alldata_I, alldata_Q, [GE, GF, EF], vote_maps, bins_universal, plot = 1, num_bins = nb)
print("Majority Vote Fidelity: [G,E,F,total,null]: \n", [fid_G, fid_E, fid_F, np.average([fid_G, fid_E, fid_F]), numberNull])
