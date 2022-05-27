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
# datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\fixed\signal_power_sweep_-30MHz_detuned\2022-05-25_0001_pwr_swp_0dB_att_Amp_0__.ddh5'

#with 50MHz bp filter:
#amp off
# datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/2022-05-26_0006_pwr_swp_0dB_att_Amp_0__.ddh5'
#amp on
# datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/2022-05-26_0007_pwr_swp_0dB_att_Amp_1__.ddh5'

#without 50MHz bandpass
# datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/lsb/2022-05-26_0002_pwr_swp_0dB_att_Amp_1__.ddh5'

#mod at +50 instead of -50
# datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\raw_data\without_50MHz_bp\2022-05-26_0004_pwr_swp_0dB_att_mod+50_Amp_1__.ddh5'

#back to -50, but switching to the upper sideband of the amplifier
# datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb/2022-05-26_0002_pwr_swp_0dB_att_mod-50_usb_Amp_1__.ddh5'

#the best trace, using the IR mixer as a good high-pass to kick the idler out of the lsb, using signal in usb
datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb/2022-05-26_0002_pwr_swp_0dB_att_mod-50_usb_Amp_1__.ddh5'

#taken the next day, and instead of using the IR mixer to kill the idler, I chose to use lsb so that I can see the idler in the data. I also chose the signal detuning to be -25mHz, so that the idler would be at 100MHz in the data. This is demodulatable by itself! use a period of 10 instead of 20
# datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\raw_data\without_50MHz_bp\lsb\2022-05-27_0002_pwr_swp_0dB_att_mod-50_usb_Amp_1__.ddh5'

# datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/lsb/2022-05-27_0001_pwr_swp_0dB_att_mod-50_usb_Amp_0__.ddh5'

# usb, 0.3V, amp off
datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb_amp_off/power_sweep/2022-05-27_0001_pwr_swp_0dB_att_mod-50_usb_Sig_Volt_0.3_V_.ddh5'

#usb, 0.3V, amp on, 
datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb/power_sweep/2022-05-27_0001_pwr_swp_0dB_att_mod-50_usb_Sig_Volt_0.3_V_.ddh5'

#phase_sensitive
datapath = r'Z:/Data/N25_L3_SP_2/time-domain/18dB_wideband_gain/raw_data/without_50MHz_bp/usb/power_sweep/2022-05-27_0016_pwr_swp_0dB_att_mod-50_usb_Sig_Volt_1.8_V_.ddh5'

#data taken 10min later
datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\raw_data\without_50MHz_bp_phase_sensitive\amp_on\2022-05-27_0001_pwr_swp_0dB_att_mod-50_usb_Rep_0__.ddh5'

#after rotating pump phase by 90 degrees
datapath = r'Z:\Data\N25_L3_SP_2\time-domain\18dB_wideband_gain\raw_data\without_50MHz_bp_phase_sensitive\amp_on\2022-05-27_0003_pwr_swp_0dB_att_mod-50_usb_+90deg_Rep_0__.ddh5'
time, signal_arr, ref_arr = pulseUtils.Process_One_Acquisition_3_state(datapath)
#%% see if there are multiple freqs in the data
sGAvg = np.average(signal_arr[0], axis = 0)
fwindow = [0, 150]
freqs = fftfreq(4096, 1e-9)/1e6
ffilt = (freqs<fwindow[1])*(freqs>fwindow[0])
plt.plot(freqs[ffilt], 10*np.log10(np.abs(fft(sGAvg)[ffilt])))
plt.title("Spectrum before filtering")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBmV)')

#%% generate a filter to select a frequency spectrum area of the data
filt = bpf_func(50e6, 10e6, 5) #center frequency, width, order of butterworth filter
w, h = sosfreqz(filt, fs = 1e9)
ffilt2 = (w/1e6>fwindow[0])*(w/1e6<fwindow[1])
plt.plot(w[ffilt2]/1e6, 10*np.log10(np.abs(h)**2)[ffilt2])
plt.title("Applied Butterworth filter")
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude $|H|^2$ (dB')
plt.ylim(-60, 5)
#%% apply the filter, demodulate, remove offsets, and plot averages
apply_filter = 1
signal_arr_filtered = []
ref_arr_filtered = []
state_data_arr = []
state_ref_arr = []
for sdata, rdata in zip(signal_arr, ref_arr): 
    if apply_filter: 
        sdata_filt, rdata_filt = filter_all_records(sdata, rdata, filt) #bw is 10MHz
    else: 
        sdata_filt, rdata_filt = sdata, rdata
    signal_arr_filtered.append(sdata_filt)
    ref_arr_filtered.append(rdata_filt)
    demod_data, demod_ref = demod_all_records(sdata_filt, rdata_filt, period = 20)
    state_data_arr.append(demod_data)
    state_ref_arr.append(demod_ref)
    
G_data, E_data, F_data = state_data_arr
G_ref, E_ref, F_ref = state_ref_arr
# phase correction
G_I_corr, G_Q_corr, rI_trace, rQ_trace = phase_correction(*G_data, *G_ref)
E_I_corr, E_Q_corr, rI_trace, rQ_trace = phase_correction(*E_data, *E_ref)
F_I_corr, F_Q_corr, rI_trace, rQ_trace = phase_correction(*F_data, *F_ref)

G_data_off, E_data_off, F_data_off = [G_I_corr, G_Q_corr], [E_I_corr, E_Q_corr], [F_I_corr, F_Q_corr]
#remove offsets then average
G_data, E_data, F_data = [remove_offset(*data, window = [0, 40]) for data in [G_data_off, E_data_off, F_data_off]]
GAvg, EAvg, FAvg = [np.average(data, axis = 1) for data in [G_data, E_data, F_data]]

fig, ax = plt.subplots()

ax.plot(GAvg[0], GAvg[1], label = 'G')
ax.plot(EAvg[0], EAvg[1], label = 'E')
ax.plot(FAvg[0], FAvg[1], label = 'F')
ax.legend()
ax.set_aspect(1)
is_filt = {0:"off", 1: 'on'}
ax.set_title(f"Average of data with filter {is_filt[apply_filter]}")
#%%check the effect of the filter
sGAvg2 = np.average(signal_arr_filtered[0], axis = 0)
plt.plot(freqs[ffilt], 10*np.log10(np.abs(fft(sGAvg2)[ffilt])))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dBmV)')
plt.title("Spectrum after filtering")

#%% This marks the stage where the data used to be saved (without a digital filter), all the rest is linear processing and majority vote


#%% histogram the data with weight functions
GE, GF, EF = [generate_matched_weight_funcs(*data, bc = True) for data in [[G_data, E_data], [G_data, F_data], [E_data, F_data]]]
alldata = np.append(np.append(G_data, E_data, axis = 1), F_data, axis = 1)
# fig2, axs2 = plt.subplots(3, 3, figsize = (6,12))
titles = np.array([['SGE_'+state, 'SGF_'+state, 'SEF_'+state] for state in ['G', 'E', 'F']]).T.flatten()
titles_all = np.array(['SGE_all', 'SGF_all', 'SEF_all'])

scale = 0.01

h_arr = [
    weighted_histogram(WF[0], WF[1], data[0], data[1], scale = scale, plot = 0, num_bins = 300) for 
    WF, data in 
    zip(
    np.repeat([GE,GF,EF], 3, axis = 0),
    np.tile([G_data, E_data, F_data], (3,1,1,1))
    )]

h_arr_all =[
    weighted_histogram(WF[0], WF[1], data[0], data[1], scale = scale, plot = 0, num_bins = 300) for 
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
fig.tight_layout()
#%%fitting
gaussians = []
gaussian_hist = []
wfs = np.repeat([GE,GF,EF], 3, axis = 0)
avgs = np.tile([GAvg,EAvg,FAvg], (3, 1, 1))
max_fev = 10000
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
    

    
    A_x0Guess = np.dot(avgs[i][0], wfs[i][0])+np.dot(avgs[i][1], wfs[i][1])
    A_y0Guess = np.dot(avgs[i][1], wfs[i][0])-np.dot(avgs[i][0], wfs[i][1])
    A_ampGuess = np.average(np.sqrt(np.array(Ipts)**2+np.array(Qpts)**2))
    A_sxGuess = A_ampGuess/5
    # A_thetaGuess = np.average(np.angle(A_x0Guess+1j*A_y0Guess))
    A_thetaGuess = 0
    
    guessParams = [A_ampGuess, A_y0Guess, A_x0Guess, A_sxGuess]
    #do a 2D Gaussian Fit to the data
    gaussians.append(fit_2D_Gaussian(titles[i], bins, h_2d, 
                                                    guessParams,
                                                    # None,
                                                    max_fev = max_fev,
                                                    contour_line = 2, debug = 1) 
                                                    )
    gaussian_hist.append(Gaussian_2D(np.meshgrid(bins[:-1], bins[:-1]), *gaussians[-1].info_dict['popt']))
    

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

#%% Majority vote
#G_data, E_data, F_data = [G_I_corr, G_Q_corr], [E_I_corr, E_Q_corr], [F_I_corr, F_Q_corr]
states = [G_data, E_data, F_data]
alldata_I, alldata_Q = [np.append(np.append(states[0][i], states[1][i], axis = 0), states[2][i], axis = 0) for i in [0, 1]]#shape of [N_records, N_time_points]
fid_G, fid_E, fid_F, numberNull = majorityVote3State(alldata_I, alldata_Q, [GE, GF, EF], vote_maps, bins_universal, plot = 1, num_bins = 300)
print("Majority Vote Fidelity: [G,E,F,total,null]: \n", [fid_G, fid_E, fid_F, np.average([fid_G, fid_E, fid_F]), numberNull])
