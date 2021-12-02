# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:40:12 2021

@author: Ryan Kaufman

Set up function module that can assist in loading pulse sequences into AWG
and functionalizing Alazar acquiring
"""
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Normalize as Norm
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from scipy.signal import butter, sosfilt

def Process_One_Acquisition_np(filepath, plot = False, pulse_types = 2):
    
    
    return pulse
    

def Process_One_Acquisition(name, sI_c1, sI_c2, sQ_c1 ,sQ_c2, bin_start, bin_stop, hist_scale = 200, odd_only = False, even_only = False, plot = False):
    if plot: 
        fig = plt.figure(1, figsize = (12,8))
        fig.suptitle(name, fontsize = 20)
        ax1 = fig.add_subplot(221)
        ax1.set_title("I")
        ax1.plot(np.average(sI_c1, axis = 0), label = 'even records')
        ax1.plot(np.average(sI_c2, axis = 0), label = 'odd_records')
        # ax1.set_aspect(1)
        ax1.legend(loc = 'upper right')
        ax2 = fig.add_subplot(222)
        ax2.set_title("Q")
        ax2.plot(np.average(sQ_c1, axis = 0), label = 'even records')
        ax2.plot(np.average(sQ_c2, axis = 0), label = 'odd records')
        # ax2.set_aspect(1)
        ax2.legend(loc = 'upper right')
        ax3 = fig.add_subplot(223)
        ax3.set_aspect(1)
        ax3.plot(np.average(sI_c1, axis = 0), np.average(sQ_c1, axis = 0))
        ax3.plot(np.average(sI_c2, axis = 0),np.average(sQ_c2, axis = 0))
        
        #figure for difference trace
        fig2 = plt.figure(2, figsize = (12,8))
        ax21 = fig2.add_subplot(221)
        ax21.set_title("I (even-odd records)")
        ax21.plot(np.average(sI_c1-sI_c2, axis = 0), label = 'even-odd records')
        
        # ax1.set_aspect(1)
        ax22 = fig2.add_subplot(222)
        ax22.set_title("Q (even-odd records)")
        ax22.plot(np.average(sQ_c1-sQ_c2, axis = 0), label = 'even-odd records')
        
        # ax2.set_aspect(1)
        ax23 = fig2.add_subplot(223)
        ax23.set_title("Trajectories")
        ax23.set_aspect(1)
        ax23.plot(np.average(sI_c1-sI_c2, axis = 0), np.average(sQ_c1-sQ_c2, axis = 0))
        
        
        ax24 = fig2.add_subplot(224)
        ax24.set_title("magnitudes")
        ax24.plot(np.average(sI_c1-sI_c2, axis = 0)**2+np.average(sQ_c1-sQ_c2, axis = 0)**2, label = 'magnitude')
        ax4 = fig.add_subplot(224)

    fig2, ax99 = plt.subplots()
    # print(np.shape(sI_c1))
    bins_even, h_even = boxcar_histogram(fig2, ax99, bin_start, bin_stop, sI_c1, sQ_c1, Ioffset = 0, Qoffset = 0, scale = hist_scale)
    bins_odd, h_odd = boxcar_histogram(fig2, ax99, bin_start, bin_stop, sI_c2, sQ_c2, Ioffset = 0, Qoffset = 0, scale = hist_scale)
    plt.close(fig2)
    
    if plot: 
        if even_only and not odd_only: 
            print('displaying only even')
            boxcar_histogram(fig, ax4, bin_start, bin_stop, sI_c1, sQ_c1, Ioffset = 0, Qoffset = 0, scale = hist_scale)
            
        elif odd_only and not even_only: 
            print('displaying only odd')
            boxcar_histogram(fig, ax4, bin_start, bin_stop, sI_c2, sQ_c2, Ioffset = 0, Qoffset = 0, scale = hist_scale)
        else: 
            print('displaying both')
            boxcar_histogram(fig, ax4, bin_start, bin_stop, np.concatenate((sI_c1, sI_c2)), np.concatenate((sQ_c1, sQ_c2)), Ioffset = 0, Qoffset = 0, scale = hist_scale)
        plt.show()
    return bins_even, bins_odd, h_even.T, h_odd.T

def Process_One_Acquisition_3_state(name, time_vals, sI_c1, sI_c2, sI_c3, sQ_c1 ,sQ_c2, sQ_c3, hist_scale = 200, odd_only = False, even_only = False, plot = False, lpf = True, lpf_wc = 50e6, fit = False, hist_y_scale = 10, boxcar = False, bc_window = [50, 150], record_track = False):
    
    if boxcar:
        WF = np.zeros(np.size(time_vals))
        WF[bc_window[0]:bc_window[1]] = 1
        Sge = Sgf = Sef = WF
    else: 
        #weight functions denoted by Sij for telling trace i from trace j
        Sge = (np.average(sI_c1, axis = 0)-np.average(sI_c2, axis = 0))+(np.average(sQ_c1, axis = 0)-np.average(sQ_c2, axis = 0))
        
        Sgf = (np.average(sI_c1, axis = 0)-np.average(sI_c3, axis = 0))+(np.average(sQ_c1, axis = 0)-np.average(sQ_c3, axis = 0))
        
        Sef = (np.average(sI_c2, axis = 0)-np.average(sI_c3, axis = 0))+(np.average(sQ_c2, axis = 0)-np.average(sQ_c3, axis = 0))

    if lpf: 
    
        Sge = sosfilt(butter(10, lpf_wc, fs = 1e9/20, output = 'sos'), Sge)
    
        Sgf = sosfilt(butter(10, lpf_wc, fs = 1e9/20, output = 'sos'), Sgf)
    
        Sef = sosfilt(butter(10, lpf_wc, fs = 1e9/20, output = 'sos'), Sef)
        
    sI_c1_avg = np.average(sI_c1, axis = 0)
    sI_c2_avg = np.average(sI_c2, axis = 0)
    sI_c3_avg = np.average(sI_c3, axis = 0)
    
    sQ_c1_avg = np.average(sQ_c1, axis = 0)
    sQ_c2_avg = np.average(sQ_c2, axis = 0)
    sQ_c3_avg = np.average(sQ_c3, axis = 0)
    
    if plot: 
        
        fig = plt.figure(1, figsize = (12,8))
        fig.suptitle(name, fontsize = 20)
        ax1 = fig.add_subplot(221)
        ax1.set_title("I average")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_xlabel("Time (ns)")
        ax1.plot(time_vals, np.average(sI_c1, axis = 0)*1000, label = 'G_records')
        ax1.plot(time_vals,np.average(sI_c2, axis = 0)*1000, label = 'E_records')
        ax1.plot(time_vals,np.average(sI_c3, axis = 0)*1000, label = 'F_records')
        ax1.grid()
        # ax1.set_aspect(1)
        ax1.legend(loc = 'upper right')
        ax2 = fig.add_subplot(222)
        ax2.set_title("Q average")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_xlabel("Time (ns)")
        ax2.plot(time_vals,np.average(sQ_c1, axis = 0)*1000, label = 'G records')
        ax2.plot(time_vals,np.average(sQ_c2, axis = 0)*1000, label = 'E records')
        ax2.plot(time_vals,np.average(sQ_c3, axis = 0)*1000, label = 'F records')
        ax2.grid()
        # ax2.set_aspect(1)
        ax2.legend(loc = 'upper right')
        ax3 = fig.add_subplot(223)
        ax3.set_title("Trajectories")
        ax3.set_ylabel("I Voltage (mV)")
        ax3.set_xlabel("Q Voltage (mV)")
        ax3.set_aspect(1)
        ax3.plot(np.average(sI_c1, axis = 0)*1000, np.average(sQ_c1, axis = 0)*1000)
        ax3.plot(np.average(sI_c2, axis = 0)*1000,np.average(sQ_c2, axis = 0)*1000)
        ax3.plot(np.average(sI_c3, axis = 0)*1000,np.average(sQ_c3, axis = 0)*1000)
        ax3.grid()
        
        ax4 = fig.add_subplot(224)
        ax4.set_title("Weight Functions")
        ax4.plot(Sge, label = 'Wge')
        ax4.plot(Sgf, label = 'Wgf')
        ax4.plot(Sef, label = 'Wef')
        ax4.legend()
        ax4.grid()
        
        fig.tight_layout(h_pad = 1, w_pad = 1.5)
        
        fig01 = plt.figure(10, figsize = (12,8))
        fig01.suptitle(name, fontsize = 20)
        ax1 = fig01.add_subplot(111)
        ax1.set_title("Magnitude Difference between G and E")
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_xlabel("Time (ns)")
        ax1.plot(time_vals, np.sqrt(sI_c1_avg**2+sQ_c1_avg**2)*1000 - np.sqrt(sI_c2_avg**2+sQ_c2_avg**2)*1000, label = 'G_records-E_records')

        ax1.grid()
        
        fig2 = plt.figure(2, figsize = (12,8))
        
        ax11 = fig2.add_subplot(331)
        ax11.set_title("GE - G")
        ax12 = fig2.add_subplot(332)
        ax12.set_title("GE - E")
        ax13 = fig2.add_subplot(333)
        ax13.set_title("GE - F")
        
        ax21 = fig2.add_subplot(334)
        ax21.set_title("GF - G")
        ax22 = fig2.add_subplot(335)
        ax22.set_title("GF - E")
        ax23 = fig2.add_subplot(336)
        ax23.set_title("GF - F")
        
        ax31 = fig2.add_subplot(337)
        ax31.set_title("EF - G")
        ax32 = fig2.add_subplot(338)
        ax32.set_title("EF - E")
        ax33 = fig2.add_subplot(339)
        ax33.set_title("EF - F")
        
        ax11.grid()
        ax12.grid()
        ax13.grid()
        ax21.grid()
        ax22.grid()
        ax23.grid()
        ax31.grid()
        ax32.grid()
        ax33.grid()
        
        fig2.tight_layout(h_pad = 1, w_pad = 1)
    
    #using GE weights: 
    if hist_scale == None: 
        hist_scale = np.max(np.abs([sI_c1_avg, sQ_c1_avg]))*1.2
        hist_scale1 = np.max(np.abs([sI_c1_avg, sQ_c1_avg]))*1.2
        hist_scale2 = hist_scale1
        hist_scale3 = hist_scale1
    else: 
        hist_scale1 = hist_scale
        hist_scale2 = hist_scale
        hist_scale3 = hist_scale
    # hist_scale2 = np.max(np.abs([sI_c2_avg, sQ_c2_avg]))*1.2
    # hist_scale3 = np.max(np.abs([sI_c3_avg, sQ_c3_avg]))*1.2
    
    #GE weights
    bins_GE_G, h_GE_G, I_GE_G_pts, Q_GE_G_pts = weighted_histogram(fig2, ax11, Sge, sI_c1, sQ_c1, scale = hist_scale1, record_track = record_track)
    bins_GE_E, h_GE_E, I_GE_E_pts, Q_GE_E_pts = weighted_histogram(fig2, ax12, Sge, sI_c2, sQ_c2, scale = hist_scale2, record_track = record_track)
    bins_GE_F, h_GE_F, I_GE_F_pts, Q_GE_F_pts = weighted_histogram(fig2, ax13, Sge, sI_c3, sQ_c3, scale = hist_scale3, record_track = record_track)
    
    #using GF weights:
    bins_GF_G, h_GF_G, I_GF_G_pts, Q_GF_G_pts = weighted_histogram(fig2, ax21, Sgf, sI_c1, sQ_c1, scale = hist_scale1, record_track = False)
    bins_GF_E, h_GF_E, I_GF_E_pts, Q_GF_E_pts = weighted_histogram(fig2, ax22, Sgf, sI_c2, sQ_c2, scale = hist_scale2, record_track = False)
    bins_GF_F, h_GF_F, I_GF_F_pts, Q_GF_F_pts = weighted_histogram(fig2, ax23, Sgf, sI_c3, sQ_c3, scale = hist_scale3, record_track = False)
    
    #using EF weights:
    bins_EF_G, h_EF_G, I_EF_G_pts, Q_EF_G_pts = weighted_histogram(fig2, ax31, Sef, sI_c1, sQ_c1, scale = hist_scale1, record_track = False)
    bins_EF_E, h_EF_E, I_EF_E_pts, Q_EF_E_pts = weighted_histogram(fig2, ax32, Sef, sI_c2, sQ_c2, scale = hist_scale2, record_track = False)
    bins_EF_F, h_EF_F, I_EF_F_pts, Q_EF_F_pts = weighted_histogram(fig2, ax33, Sef, sI_c3, sQ_c3, scale = hist_scale3, record_track = False)
    
    if fit: 
        
        I_G = sI_c1
        Q_G = sQ_c1
        I_E = sI_c2
        Q_E = sQ_c2
        I_F = sI_c3
        Q_F = sQ_c3
        
        I_G_avg = np.average(I_G, axis = 0)
        I_E_avg = np.average(I_E, axis = 0)
        I_F_avg = np.average(I_F, axis = 0)
        
        Q_G_avg = np.average(Q_G, axis = 0)
        Q_E_avg = np.average(Q_E, axis = 0)
        Q_F_avg = np.average(Q_F, axis = 0)
        
        
        G_x0Guess = np.max(I_G_avg)
        G_y0Guess = np.max(Q_G_avg)
        G_ampGuess = np.average(np.sqrt(I_G_avg**2+Q_G_avg**2))
        G_sxGuess = hist_scale/2
        G_syGuess = hist_scale/2
        G_thetaGuess = np.average(np.angle(I_G_avg+1j*Q_G_avg))
        G_offsetGuess = 0
        
        E_x0Guess = np.max(I_E_avg)
        E_y0Guess = np.max(Q_E_avg)
        E_ampGuess = np.average(np.sqrt(I_E_avg**2+Q_E_avg**2))
        E_sxGuess = hist_scale/2
        E_syGuess = hist_scale/2
        E_thetaGuess = np.average(np.angle(I_E_avg+1j*Q_E_avg))
        E_offsetGuess = 0
        
        F_x0Guess = np.max(I_F_avg)
        F_y0Guess = np.max(Q_F_avg)
        F_ampGuess = np.average(np.sqrt(I_F_avg**2+Q_F_avg**2))
        F_sxGuess = hist_scale/2
        F_syGuess = hist_scale/2
        F_thetaGuess = np.average(np.angle(I_F_avg+1j*Q_F_avg))
        F_offsetGuess = 0
        
        guessParams = [[G_ampGuess, G_x0Guess, G_y0Guess, G_sxGuess, G_thetaGuess],
                       [E_ampGuess, E_x0Guess, E_y0Guess, E_sxGuess, E_thetaGuess], 
                       [F_ampGuess, F_x0Guess, F_y0Guess, F_sxGuess, F_thetaGuess]]
        
        print(guessParams)
        ########
        max_fev = 10000
        GE_G_fit = fit_2D_Gaussian('GE_G_fit', bins_GE_G, h_GE_G, 
                                                    guessParams[0],
                                                    max_fev = max_fev,
                                                    contour_line = 4)
        GE_G_fit_h = Gaussian_2D(np.meshgrid(bins_GE_G[:-1], bins_GE_G[:-1]), *GE_G_fit.info_dict['popt'])
        GE_G_fit_h_norm = np.copy(GE_G_fit_h/np.sum(GE_G_fit_h))
        ########
        GE_E_fit = fit_2D_Gaussian('GE_E_fit', bins_GE_E, h_GE_E, 
                                                    guessParams[1],
                                                    max_fev = max_fev,
                                                    contour_line = 4)
        GE_E_fit_h = Gaussian_2D(np.meshgrid(bins_GE_E[:-1], bins_GE_E[:-1]), *GE_E_fit.info_dict['popt'])
        GE_E_fit_h_norm = np.copy(GE_E_fit_h/np.sum(GE_E_fit_h))
        ########
        GF_G_fit = fit_2D_Gaussian('GF_G_fit', bins_GF_G, h_GF_G,
                                                guessParams[0],
                                                max_fev = max_fev,
                                                contour_line = 4)
        GF_G_fit_h = Gaussian_2D(np.meshgrid(bins_GF_G[:-1], bins_GF_G[:-1]), *GF_G_fit.info_dict['popt'])
        GF_G_fit_h_norm = np.copy(GF_G_fit_h/np.sum(GF_G_fit_h))
        
        GF_F_fit = fit_2D_Gaussian('GF_F_fit', bins_GF_F, h_GF_F,
                                                guessParams[2],
                                                max_fev = max_fev,
                                                contour_line = 4)
        GF_F_fit_h = Gaussian_2D(np.meshgrid(bins_GF_F[:-1], bins_GF_F[:-1]), *GF_F_fit.info_dict['popt'])
        GF_F_fit_h_norm = np.copy(GF_F_fit_h/np.sum(GF_F_fit_h))
        
        EF_E_fit = fit_2D_Gaussian('EF_E_fit', bins_EF_E, h_EF_E,
                                                guessParams[2],
                                                max_fev = max_fev,
                                                contour_line = 4)
        EF_E_fit_h = Gaussian_2D(np.meshgrid(bins_EF_E[:-1], bins_EF_E[:-1]), *EF_E_fit.info_dict['popt'])
        EF_E_fit_h_norm = np.copy(EF_E_fit_h/np.sum(EF_E_fit_h))
        
        EF_F_fit = fit_2D_Gaussian('EF_F_fit', bins_EF_F, h_EF_F,
                                                guessParams[2],
                                                max_fev = max_fev,
                                                contour_line = 4)
        EF_F_fit_h = Gaussian_2D(np.meshgrid(bins_EF_F[:-1], bins_EF_F[:-1]), *EF_F_fit.info_dict['popt'])
        EF_F_fit_h_norm = np.copy(EF_F_fit_h/np.sum(EF_F_fit_h))
        
        GE_is_G = hist_discriminant(GE_G_fit_h, GE_E_fit_h)
        GE_is_E = np.logical_not(GE_is_G)
        
        GF_is_G = hist_discriminant(GF_G_fit_h, GF_F_fit_h)
        GF_is_F = np.logical_not(GF_is_G)
        
        EF_is_E = hist_discriminant(EF_E_fit_h, EF_F_fit_h)
        EF_is_F = np.logical_not(EF_is_E)

        if plot: 
            fig3, axs = plt.subplots(2, 3, figsize = (12,8))
            viridis = cm.get_cmap('magma', 256)
            newcolors = viridis(np.linspace(0, 1, 256))
            gray = np.array([0.1, 0.1, 0.1, 0.1])
            newcolors[128-5: 128+5] = gray
            newcmp = ListedColormap(newcolors)

            ax1 = axs[0,0]
            ax2 = axs[0,1]
            ax3 = axs[0,2]
            
            ax1.set_title("Sge - inputs G and E")
            ax1.pcolormesh(bins_GE_G, bins_GE_G, h_GE_G+h_GE_E)
            
            ax2.set_title("Sgf - inputs G and F")
            ax2.pcolormesh(bins_GF_G, bins_GF_F, h_GF_G+h_GF_F)
            
            ax3.set_title("Sef - inputs E and F")
            ax3.pcolormesh(bins_EF_E, bins_EF_F, h_EF_E+h_EF_F)
            
#*(GE_is_G-1/2)
            scale = np.max((GE_G_fit_h+GE_E_fit_h))
            pc1 = axs[1,0].pcolormesh(bins_GE_G, bins_GE_G, (GE_G_fit_h+GE_E_fit_h)*(GE_is_G-1/2)/scale*5, cmap = newcmp, vmin = -1, vmax = 1)
            plt.colorbar(pc1, ax = axs[1,0],fraction=0.046, pad=0.04)
            GE_G_fit.plot_on_ax(axs[1,0])
            axs[1,0].add_patch(GE_G_fit.sigma_contour())
            GE_E_fit.plot_on_ax(axs[1,0])
            axs[1,0].add_patch(GE_E_fit.sigma_contour())
            
            scale = np.max((GF_G_fit_h+GF_F_fit_h))
            pc2 = axs[1,1].pcolormesh(bins_GE_G, bins_GE_G, (GF_is_G-1/2)*(GF_G_fit_h+GF_F_fit_h)/scale*5, cmap = newcmp, vmin = -1, vmax = 1)
            plt.colorbar(pc1, ax = axs[1,1],fraction=0.046, pad=0.04)
            GF_G_fit.plot_on_ax(axs[1,1])
            axs[1,1].add_patch(GF_G_fit.sigma_contour())
            GF_F_fit.plot_on_ax(axs[1,1])
            axs[1,1].add_patch(GF_F_fit.sigma_contour())
            
            scale = np.max((EF_E_fit_h+EF_F_fit_h))
            pc3 = axs[1,2].pcolormesh(bins_GE_G, bins_GE_G, (EF_is_E-1/2)*(EF_E_fit_h+EF_F_fit_h)/scale*5, cmap = newcmp, vmin = -1, vmax = 1)
            plt.colorbar(pc1, ax = axs[1,2],fraction=0.046, pad=0.04)
            EF_E_fit.plot_on_ax(axs[1,2])
            axs[1,2].add_patch(EF_E_fit.sigma_contour())
            EF_F_fit.plot_on_ax(axs[1,2])
            axs[1,2].add_patch(EF_F_fit.sigma_contour())
            
            fig3.tight_layout(h_pad = 0.1, w_pad = 1)
            
            for ax in np.array(axs).flatten(): 
                ax.set_aspect(1)
                ax.grid()
        
        
        print(np.shape)
        #classify the records - done for each weight function
        results = []
        GE_results = []
        GF_results = []
        EF_results = []
        all_I = np.vstack((sI_c1, sI_c2, sI_c3))
        all_Q = np.vstack((sQ_c1, sQ_c2, sQ_c3))
        # print("all_I shape: ", np.shape(all_I))
        # print(np.shape(list(zip(sI_c1, sQ_c1))))
        for G_record in list(zip(all_I, all_Q)): 
            It, Qt = G_record[0], G_record[1]
            
            #GE weights
            Sge_I = np.dot(Sge, It)
            Sge_Q = np.dot(Sge, Qt)
            
            Iloc = np.digitize(Sge_I, bins_GE_G)
            Qloc = np.digitize(Sge_Q, bins_GE_G)
            
            #if 1 it's G
            Sge_result = GE_is_G[Iloc, Qloc]
            
            #GF weights
            Sgf_I = np.dot(Sgf, It)
            Sgf_Q = np.dot(Sgf, Qt)
            
            Iloc = np.digitize(Sgf_I, bins_GF_G)
            Qloc = np.digitize(Sgf_Q, bins_GF_G)
            
            #if 1 it's G
            Sgf_result = GF_is_G[Iloc, Qloc]
            
            #EF weights
            Sef_I = np.dot(Sef, It)
            Sef_Q = np.dot(Sef, Qt)
            
            Iloc = np.digitize(Sef_I, bins_EF_E)
            Qloc = np.digitize(Sef_Q, bins_EF_E)
            
            #if 1 it's E
            Sef_result = EF_is_E[Iloc, Qloc]
            
            if Sge_result*Sgf_result: 
                result = 1 #G
            elif not Sge_result and Sef_result: 
                result = 2 #E
            else: 
                result = 3 #F
            
            results.append(result)
            GE_results.append(Sge_result)
            GF_results.append(Sgf_result)
            EF_results.append(Sef_result)
            
        results = np.array(results)
        
        #rescale so G-> 1, E-> 2, F -> 3
        GE_results = np.logical_not(np.array(GE_results))+1
        GF_results = np.logical_not(np.array(GF_results))*2+1
        EF_results = np.logical_not(np.array(EF_results))+2
        div1 = np.shape(sI_c1)[0]
        numRecords = 3*div1
        # print(div1)
        correct_classifications = np.append(np.append(np.ones(div1), 2*np.ones(div1)), 3*np.ones(div1))

        fig, ax = plt.subplots(5,1, figsize = (4, 8))
        
        viridisBig = cm.get_cmap('viridis', 512)
        _cmap = ListedColormap(viridisBig(np.linspace(0, 1, 256)))
        
        scale = Norm(vmin = 1, vmax = 3)
        
        
        ax[0].set_title("Correct classifications")
        ax[0].imshow([correct_classifications, correct_classifications], interpolation = 'none', cmap = _cmap, norm = scale)
        
        ax[1].set_title("GE classifications")
        ax[1].imshow([GE_results,GE_results], interpolation = 'none', cmap = _cmap, norm = scale)
        
        ax[2].set_title("GF classifications")
        ax[2].imshow([GF_results,GF_results], interpolation = 'none', cmap = _cmap, norm = scale)
        
        ax[3].set_title("EF classifications")
        ax[3].imshow([EF_results,EF_results], interpolation = 'none', cmap = _cmap, norm = scale)
        
        ax[4].set_title("Final classifications")
        ax[4].get_yaxis().set_ticks([])
        ax[4].set_label("Record number")
        ax[4].imshow([results, results], interpolation = 'none', cmap = _cmap, norm = scale)
        ax[4].set_aspect(1000)
        
        for axi in ax: 
            axi.get_yaxis().set_ticks([])
            axi.set_aspect(1000)
        right = correct_classifications==results
        # ax[2].imshow([right, right], interpolation = 'none')
        # ax[2].set_aspect(1000)
        fig.tight_layout(h_pad = 1, w_pad = 1)
    
        fidelity = np.round(np.sum(correct_classifications==results)/numRecords, 3)
        print("checking sum: ", np.max(correct_classifications[2*div1:-1]==results[2*div1:-1]))
        G_fidelity = np.round(np.sum(correct_classifications[0:div1]==results[0:div1])/div1, 3)
        E_fidelity = np.round(np.sum(correct_classifications[div1:2*div1]==results[div1:2*div1])/div1, 3)
        F_fidelity = np.round(np.sum(correct_classifications[2*div1:-1]==results[2*div1:-1])/div1, 3)
        

        return G_fidelity, E_fidelity, F_fidelity, fidelity
    
    
def boxcar_histogram(fig, ax,start_pt, stop_pt, sI, sQ, Ioffset = 0, Qoffset = 0, scale = 1, num_bins = 100):
    I_bground = Ioffset
    Q_bground = Qoffset
    # print(I_bground, Q_bground)
    I_pts = []
    Q_pts = []
    for I_row, Q_row in zip(sI, sQ): 
        I_pts.append(np.average(I_row[start_pt:stop_pt]-I_bground))
        Q_pts.append(np.average(Q_row[start_pt:stop_pt]-Q_bground))
    # plt.imshow(np.histogram2d(np.array(I_pts), np.array(Q_pts))[0])
    divider = make_axes_locatable(ax)
    ax.set_aspect(1)
    bins = np.linspace(-1,1, num_bins)*scale
    (h, xedges, yedges, im) = ax.hist2d(I_pts, Q_pts, bins = [bins, bins])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax.hexbin(I_pts, Q_pts, extent = np.array([-1,1,-1,1])*scale)
    # ax.set_xticks(np.array([-100,-75,-50,-25,0,25,50,75,100])*scale/100)
    # ax.set_yticks(np.array([-100,-75,-50,-25,0,25,50,75,100])*scale/100)
    ax.grid()
    
    return bins, h

def weighted_histogram(fig, ax, weight_function_arr, sI, sQ, scale = 1, num_bins = 100, record_track = False): 
    I_pts = []
    Q_pts = []
    print("size check: ", np.shape(sI))
    print("weights: ", np.shape(weight_function_arr))
    for I_row, Q_row in zip(sI, sQ): 
        I_pts.append(np.dot(I_row, weight_function_arr))
        Q_pts.append(np.dot(Q_row, weight_function_arr))
    # plt.imshow(np.histogram2d(np.array(I_pts), np.array(Q_pts))[0])
    divider = make_axes_locatable(ax)
    ax.set_aspect(1)
    bins = np.linspace(-1,1, num_bins)*scale
    (h, xedges, yedges, im) = ax.hist2d(I_pts, Q_pts, bins = [bins, bins])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    # ax.hexbin(I_pts, Q_pts, extent = np.array([-1,1,-1,1])*scale)
    # ax.set_xticks(np.array([-100,-75,-50,-25,0,25,50,75,100])*scale/100)
    # ax.set_yticks(np.array([-100,-75,-50,-25,0,25,50,75,100])*scale/100)
    if record_track: 
        fig2, ax2 = plt.subplots()
        ax2.set_title("Record Tracking: Demodulated signals")
        ax2.set_xlabel("time (~us)")
        ax2.set_ylabel("$\phi(t)$")
        unwrapped_phases = np.mod(np.unwrap(np.arctan(np.array(I_pts[0:500])/np.array(Q_pts[0:500])), period = np.pi), 2*np.pi)
        ax2.plot(np.arange(500)*500, unwrapped_phases, '.', label = "phi(t)")
        print("Average phase difference between records: ", np.average(np.diff(unwrapped_phases))/np.pi*180, ' degrees')
        # ax2.hlines(-12*np.pi, 0, 20000)
    
    return bins, h, I_pts, Q_pts
'''
def Gaussian_2D(M,amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = M
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g
'''
def Gaussian_2D(M,amplitude, xo, yo, sigma, theta):
    x, y = M
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
    b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
    c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g

class Gaussian_info: 
    def __init__(self): 
        self.info_dict = {}
    def print_info(self):
        for key, val in self.info_dict.items(): 
            if key == 'popt':
                pass
            elif key == 'pcov':
                pass
            else: 
                print(key, ': ', val)
                
    def __sub__(self, other_GC):
        sub_class = Gaussian_info()
        for key, val in self.info_dict.items(): 
            # print(key, val)
            if type(val) == np.float64: 
                sub_class.info_dict[key] = val - other_GC.info_dict[key]
            else: 
                sub_class.info_dict[key] = None
        return sub_class
    
    def center_vec(self): 
        return np.array([self.info_dict['x0'], self.info_dict['y0']])
    def plot_on_ax(self, ax, displacement = np.array([0,0]), color = 'white'): 
        ax.annotate("", xy=self.center_vec(), xytext=(0, 0), arrowprops=dict(arrowstyle = '->', lw = 3, color = color))
    def plot_array(self):
        return Gaussian_2D(*self.info_dict['popt'])
    def sigma_contour(self): 
        x0, y0 = self.center_vec()
        sx = self.info_dict['sigma_x']
        sy = self.info_dict['sigma_y']
        angle = self.info_dict['theta']
        return Ellipse((x0, y0), sx, sy, angle = angle/(2*np.pi)*360, 
                       fill = False, 
                       ls = '--',
                       color = 'red',
                       lw = 2)
        

def fit_2D_Gaussian(name, 
                    bins, 
                    h_arr, 
                    guessParams, 
                    max_fev = 10000, 
                    contour_line = 3): 
    print("fitting with maxfev = ", max_fev)
    X, Y = np.meshgrid(bins[0:-1], bins[0:-1])
    resh_size = np.shape(X)
    xdata, ydata= np.vstack((X.ravel(), Y.ravel())), h_arr.ravel()
    # print('xdata_shape: ', np.shape(xdata))
    # print("y shape: ",np.shape(ydata))
    print("running curve_fit")
    #,amplitude, xo, yo, sigma_x, sigma_y, theta
    bounds = [[0,np.min(bins), np.min(bins), 0, 0, 0],
              [np.max(h_arr), np.max(bins), np.max(bins), np.max(bins), np.max(bins), np.pi/2]]
    popt, pcov = curve_fit(Gaussian_2D, xdata, ydata, p0 = guessParams, maxfev = max_fev)
    GC = Gaussian_info()
    GC.info_dict['name'] = name
    GC.info_dict['canvas'] = xdata 
    GC.info_dict['amplitude'] = popt[0]
    GC.info_dict['x0'] = popt[1]
    GC.info_dict['y0'] = popt[2]
    GC.info_dict['sigma_x'] = np.abs(popt[3])
    GC.info_dict['sigma_y'] = np.abs(popt[3])
    GC.info_dict['theta'] = popt[4]
    GC.info_dict['popt'] = popt
    GC.info_dict['pcov'] = pcov
    GC.info_dict['contour'] = get_contour_line(X, Y, Gaussian_2D(xdata, *popt).reshape(resh_size), contour_line = contour_line)
    
    return GC

def get_contour_line(cont_x, cont_y, contour_arr, contour_line = 3):
    fig = plt.figure()
    contour_map = plt.contour(cont_x, cont_y, contour_arr)
    plt.close(fig)
    v = contour_map.collections[contour_line].get_paths()[0].vertices
    plot_y, plot_x = v[:,1], v[:,0]
    return plot_x, plot_y

def extract_2pulse_histogram_from_filepath(datapath, plot = False, bin_start = 55, bin_stop = 150, hist_scale = None, even_only = False, odd_only = False, numRecords = 3840*2, IQ_offset = (0,0)): 
    I_offset, Q_offset = IQ_offset
    dd = all_datadicts_from_hdf5(datapath)['data']
    
    time_unit = dd['time']['unit']
    
    # print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    I_plus = dd['I_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    I_minus = dd['I_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    
    Q_plus = dd['Q_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    Q_minus = dd['Q_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    
    # print(np.size(I_minus))
    
    #averages
    I_plus_avg = np.average(I_plus, axis = 0)
    I_minus_avg = np.average(I_minus, axis = 0)
    Q_plus_avg = np.average(Q_plus, axis = 0)
    Q_minus_avg = np.average(Q_minus, axis = 0)
    
    if hist_scale == None: 
        hist_scale = 2*np.max(np.array([I_plus_avg, I_minus_avg, Q_plus_avg, Q_minus_avg]))
    
    #re-weave the data back into it's original pre-saved form
    
    bins_even, bins_odd, h_even, h_odd = Process_One_Acquisition(datapath.split('/')[-1].split('\\')[-1], I_plus, I_minus, Q_plus, Q_minus, bin_start, bin_stop, hist_scale = hist_scale, even_only = even_only, odd_only = odd_only, plot = plot)
    
    Plus_x0Guess = np.average(np.average(I_plus_avg[bin_start:bin_stop]))
    Plus_y0Guess = np.average(np.average(Q_plus_avg[bin_start:bin_stop]))
    Plus_ampGuess = np.max(h_even)
    Plus_sxGuess = np.max(bins_even)/5
    Plus_syGuess = Plus_sxGuess
    Plus_thetaGuess = 0
    Plus_offsetGuess = 0
    
    Minus_x0Guess = np.average(np.average(I_minus_avg[bin_start:bin_stop]))
    Minus_y0Guess = np.average(np.average(Q_minus_avg[bin_start:bin_stop]))
    Minus_ampGuess = np.max(h_even)
    Minus_sxGuess = np.max(bins_even)/5
    Minus_syGuess = Minus_sxGuess
    Minus_thetaGuess = 0
    Minus_offsetGuess = 0
    
    guessParams = [[Plus_ampGuess, Plus_x0Guess, Plus_y0Guess, Plus_sxGuess, Plus_syGuess, Plus_thetaGuess],
                   [Minus_ampGuess, Minus_x0Guess, Minus_y0Guess, Minus_sxGuess, Minus_syGuess, Minus_thetaGuess]]
    
    return bins_even, bins_odd, h_even, h_odd, guessParams

def extract_3pulse_phase_differences_from_filepath(datapath, numRecords = 3840*2, window = [0, -1], bc_window = [50, 150], scale = 2):
    dd = all_datadicts_from_hdf5(datapath)['data']

    offset = window[0]
    rtrim = window[-1]
    
    time_unit = dd['time']['unit']
    I_offset, Q_offset = 0,0
    # print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))

    I_G = dd['I_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_E = dd['I_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_F = dd['I_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    
    Q_G = dd['Q_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_E = dd['Q_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_F = dd['Q_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset

    #averages
    I_G_avg = np.average(I_G, axis = 0)
    I_E_avg = np.average(I_E, axis = 0)
    I_F_avg = np.average(I_F, axis = 0)
    
    Q_G_avg = np.average(Q_G, axis = 0)
    Q_E_avg = np.average(Q_E, axis = 0)
    Q_F_avg = np.average(Q_F, axis = 0)
    
    WF = np.zeros(np.size(time_vals[0]))
    WF[bc_window[0]:bc_window[1]] = 1
    Sge = Sgf = Sef = WF
    fig2, ax11 = plt.subplots()
    
    bins_GE_G, h_GE_G, I_pts, Q_pts = weighted_histogram(fig2, ax11, Sge, I_G, Q_G, scale = scale, record_track = True)

    fig2, ax2 = plt.subplots()
    ax2.set_title("Record Tracking")
    ax2.set_xlabel("time (~us)")
    ax2.set_ylabel("$\phi(t)$")
    unwrapped_phases = np.unwrap(np.arctan(np.array(I_pts[0:500])/np.array(Q_pts[0:500])), period = np.pi)
    ax2.plot(np.arange(500)*500, unwrapped_phases, '.', label = "phi(t)")
    print("Average phase difference between records: ", np.average(np.diff(unwrapped_phases))/np.pi*180, ' degrees')
    ax2.hlines(-12*np.pi, 0, 20000)
    # ax2.set_aspect(1)
    # ax2.plot(Q_pts[0:500], '.', label = "Q")
    ax2.grid()
    
    return np.average(np.diff(unwrapped_phases))/np.pi*180

def extract_3pulse_noise_from_filepath(datapath, numRecords = 3840*2, window = [0, -1]):
    dd = all_datadicts_from_hdf5(datapath)['data']
    
    offset = window[0]
    rtrim = window[-1]
    
    time_unit = dd['time']['unit']
    I_offset, Q_offset = 0,0
    # print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))

    I_G = dd['I_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_E = dd['I_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_F = dd['I_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    
    Q_G = dd['Q_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_E = dd['Q_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_F = dd['Q_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset

    #averages
    I_G_avg = np.average(I_G, axis = 0)
    I_E_avg = np.average(I_E, axis = 0)
    I_F_avg = np.average(I_F, axis = 0)
    
    Q_G_avg = np.average(Q_G, axis = 0)
    Q_E_avg = np.average(Q_E, axis = 0)
    Q_F_avg = np.average(Q_F, axis = 0)
    print(np.shape(I_G))
    return np.sqrt(np.var(np.sqrt(I_G[:, offset: rtrim]**2+Q_G[:, offset: rtrim]**2)))

def extract_3pulse_pwr_from_filepath(datapath, numRecords = 3840*2, window = [0, -1]):
    dd = all_datadicts_from_hdf5(datapath)['data']
    
    offset = window[0]
    rtrim = window[-1]
    
    time_unit = dd['time']['unit']
    I_offset, Q_offset = 0,0
    # print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))

    I_G = dd['I_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_E = dd['I_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_F = dd['I_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    
    Q_G = dd['Q_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_E = dd['Q_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_F = dd['Q_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset

    #averages
    I_G_avg = np.average(I_G, axis = 0)
    I_E_avg = np.average(I_E, axis = 0)
    I_F_avg = np.average(I_F, axis = 0)
    
    Q_G_avg = np.average(Q_G, axis = 0)
    Q_E_avg = np.average(Q_E, axis = 0)
    Q_F_avg = np.average(Q_F, axis = 0)

    return np.average(np.sqrt(I_G_avg**2+Q_G_avg**2)[offset:rtrim]), np.average(np.sqrt(I_E_avg**2+Q_E_avg**2)[offset:rtrim]), np.average(np.sqrt(I_F_avg**2+Q_F_avg**2)[offset:rtrim])


def extract_3pulse_histogram_from_filepath(datapath, plot = False, hist_scale = None, numRecords = 3840*2, IQ_offset = (0,0), fit = False, lpf = True, lpf_wc = 50e6, boxcar = False, bc_window = [50, 150], record_track = True, tuneup_plots = True):
    I_offset, Q_offset = IQ_offset
    dd = all_datadicts_from_hdf5(datapath)['data']
    print("dd keys",dd.keys())
    time_unit = dd['time']['unit']
    
    # print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))

    I_G = dd['I_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_E = dd['I_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    I_F = dd['I_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-I_offset
    
    Q_G = dd['Q_G']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_E = dd['Q_E']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset
    Q_F = dd['Q_F']['values'].reshape((numRecords//3, np.size(dd['time']['values'])//(numRecords//3)))-Q_offset

    #averages
    I_G_avg = np.average(I_G, axis = 0)
    I_E_avg = np.average(I_E, axis = 0)
    I_F_avg = np.average(I_F, axis = 0)
    
    Q_G_avg = np.average(Q_G, axis = 0)
    Q_E_avg = np.average(Q_E, axis = 0)
    Q_F_avg = np.average(Q_F, axis = 0)

    return Process_One_Acquisition_3_state(datapath.split('/')[-1].split('\\')[-1], time_vals[0], I_G, I_E, I_F, Q_G, Q_E, Q_F,hist_scale = hist_scale, plot = plot, fit = fit, lpf = lpf, lpf_wc = lpf_wc, boxcar = boxcar, bc_window = bc_window, record_track = record_track) 
        
def get_normalizing_voltage_from_filepath(amp_off_filepath, plot = False, hist_scale = None, records_per_pulsetype = 3870*2): 
    
    bins_even, bins_odd, h_even, h_odd, guessParam = extract_2pulse_histogram_from_filepath(amp_off_filepath, 
                                                                                               odd_only = 0, 
                                                                                               numRecords = int(3840*2), 
                                                                                               IQ_offset = (0,0), 
                                                                                               plot = plot, 
                                                                                               hist_scale = hist_scale)
    
    amp_off_even_fit = fit_2D_Gaussian('amp_off_even', bins_even, h_even, 
                                            guessParam[0],
                                            max_fev = 1000,
                                            contour_line = 2)
    amp_off_odd_fit = fit_2D_Gaussian('amp_off_odd', bins_odd, h_odd,
                                            guessParam[1],
                                            max_fev = 1000,
                                            contour_line = 2)
    even_fit = amp_off_even_fit
    odd_fit = amp_off_odd_fit
    
    histogram_data_fidelity = 1-1/2*np.sum(np.sqrt((h_odd/records_per_pulsetype)*(h_even/records_per_pulsetype)))
            
    bins_fine = np.linspace(np.min([bins_even, bins_odd]), np.max([bins_even, bins_odd]), 1000)
    
    even_fit_h = Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *even_fit.info_dict['popt'])/(2*np.pi*even_fit.info_dict['amplitude']*even_fit.info_dict['sigma_x']*even_fit.info_dict['sigma_y'])
    
    odd_fit_h = Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *odd_fit.info_dict['popt'])/(2*np.pi*odd_fit.info_dict['amplitude']*odd_fit.info_dict['sigma_x']*odd_fit.info_dict['sigma_y'])
    
    fit_fidelity = 1-1/2*np.sum(np.sqrt(np.abs(even_fit_h)/np.sum(even_fit_h)*np.abs(odd_fit_h)/np.sum(odd_fit_h)))
    print(f"Histogram data fidelity: {histogram_data_fidelity}\nFit fidelity: {fit_fidelity}")
    if plot: 
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_even, bins_even, h_even)
        amp_off_even_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_even_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
        
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_odd, bins_odd, h_odd)
        amp_off_odd_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_odd_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
    
    amp_off_voltage = np.average([np.linalg.norm(amp_off_odd_fit.center_vec()), np.linalg.norm(amp_off_even_fit.center_vec())])*1000
    
    return amp_off_voltage

def get_IQ_offset_from_filepath(amp_off_filepath, plot = False, hist_scale = None, records_per_pulsetype = 3840*2): 
    
    bins_even, bins_odd, h_even, h_odd, guessParam = extract_2pulse_histogram_from_filepath(amp_off_filepath, 
                                                                                               odd_only = 0, 
                                                                                               numRecords = records_per_pulsetype, 
                                                                                               IQ_offset = (0,0), 
                                                                                               plot = True, 
                                                                                               hist_scale = hist_scale)
    
    amp_off_even_fit = fit_2D_Gaussian('amp_off_even', bins_even, h_even, 
                                            guessParam[0],
                                            max_fev = 1000,
                                            contour_line = 2)
    amp_off_odd_fit = fit_2D_Gaussian('amp_off_odd', bins_odd, h_odd,
                                            guessParam[1],
                                            max_fev = 1000,
                                            contour_line = 2)
    even_fit = amp_off_even_fit
    odd_fit = amp_off_odd_fit
    
    histogram_data_fidelity = 1-1/2*np.sum(np.sqrt((h_odd/records_per_pulsetype)*(h_even/records_per_pulsetype)))
            
    bins_fine = np.linspace(np.min([bins_even, bins_odd]), np.max([bins_even, bins_odd]), 1000)
    
    even_fit_h = Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *even_fit.info_dict['popt'])/(2*np.pi*even_fit.info_dict['amplitude']*even_fit.info_dict['sigma_x']*even_fit.info_dict['sigma_y'])
    
    odd_fit_h = Gaussian_2D(np.meshgrid(bins_fine, bins_fine), *odd_fit.info_dict['popt'])/(2*np.pi*odd_fit.info_dict['amplitude']*odd_fit.info_dict['sigma_x']*odd_fit.info_dict['sigma_y'])
    
    fit_fidelity = 1-1/2*np.sum(np.sqrt(np.abs(even_fit_h)/np.sum(even_fit_h)*np.abs(odd_fit_h)/np.sum(odd_fit_h)))
    print(f"Histogram data fidelity: {histogram_data_fidelity}\nFit fidelity: {fit_fidelity}")
    if plot: 
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_even, bins_even, h_even)
        amp_off_even_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_even_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
        
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_odd, bins_odd, h_odd)
        amp_off_odd_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_odd_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
    
    offset = np.average(np.array([amp_off_odd_fit.center_vec(), amp_off_even_fit.center_vec()]), axis = 0)
    
    return offset

def hist_discriminant(h1, h2):
    #1 if in h1, 0 if in h2
    return ((h1-h2)>0)

def get_fidelity_from_filepath(filepath, plot = False, hist_scale = None, records_per_pulsetype = 3870*2): 
    
    bins_even, bins_odd, h_even, h_odd, guessParam = extract_2pulse_histogram_from_filepath(filepath, 
                                                                                               odd_only = 0, 
                                                                                               numRecords = records_per_pulsetype, 
                                                                                               IQ_offset = (0,0), 
                                                                                               plot = True, 
                                                                                               hist_scale = hist_scale)
    h_odd_norm = np.copy(h_odd/np.sum(h_odd))
    h_even_norm = np.copy(h_even/np.sum(h_even))
    
    amp_off_even_fit = fit_2D_Gaussian('amp_off_even', bins_even, h_even, 
                                            guessParam[0],
                                            max_fev = 1000,
                                            contour_line = 2)
    amp_off_odd_fit = fit_2D_Gaussian('amp_off_odd', bins_odd, h_odd,
                                            guessParam[1],
                                            max_fev = 1000,
                                            contour_line = 2)
    even_fit = amp_off_even_fit
    odd_fit = amp_off_odd_fit
    
    even_fit_h = Gaussian_2D(np.meshgrid(bins_even[:-1], bins_even[:-1]), *even_fit.info_dict['popt'])
    even_fit_h_norm = np.copy(even_fit_h/np.sum(even_fit_h))
    
    odd_fit_h = Gaussian_2D(np.meshgrid(bins_odd[:-1], bins_odd[:-1]), *odd_fit.info_dict['popt'])
    odd_fit_h_norm = np.copy(odd_fit_h/np.sum(odd_fit_h))
    
    is_even = hist_discriminant(even_fit_h, odd_fit_h)
    is_odd = np.logical_not(is_even)
    
    #debugging
    # print(np.sum(h_odd), np.sum(h_even))
    # print(np.sum(h_odd_norm), np.sum(h_even_norm))
    # print('fid sums', np.sum(h_odd_norm[is_even]), np.sum(h_even_norm[is_odd]))
    
    plt.pcolormesh(bins_odd, bins_odd, h_odd_norm)
    plt.colorbar()
    
    data_fidelity = 1-np.sum(h_odd_norm[is_even], dtype = "float64")-np.sum(h_even_norm[is_odd], dtype = "float64")
    fit_fidelity = 1-np.sum(odd_fit_h_norm[is_even], dtype = "float64")-np.sum(even_fit_h_norm[is_odd], dtype = "float64")
    
    if plot: 
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_even, bins_even, h_even)
        amp_off_even_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_even_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
        
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_odd, bins_odd, h_odd)
        amp_off_odd_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_odd_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
        
        fig, ax = plt.subplots()
        pc = ax.pcolormesh(bins_odd, bins_odd, is_even, cmap = 'seismic')
        amp_off_odd_fit.plot_on_ax(ax)
        amp_off_even_fit.plot_on_ax(ax)
        ax.add_patch(amp_off_odd_fit.sigma_contour())
        ax.add_patch(amp_off_even_fit.sigma_contour())
        ax.set_aspect(1)
        plt.colorbar(pc)
    
    
    return data_fidelity, fit_fidelity, even_fit, odd_fit

def get_fidelity_from_filepath_3_state(filepath, plot = False, hist_scale = None, records_per_pulsetype = 2562, state_relabel = 0, bin_start = 50, bin_stop = 150, fit = False): 
    
    bins_G, bins_E, bins_F, h_G, h_E, h_F, guessParam = extract_3pulse_histogram_from_filepath(filepath, 
                                                                                               odd_only = 0, 
                                                                                               numRecords = records_per_pulsetype*3, 
                                                                                               IQ_offset = (0,0), 
                                                                                               plot = True, 
                                                                                               hist_scale = hist_scale, 
                                                                                               state_relabel = state_relabel, 
                                                                                               bin_start = bin_start, 
                                                                                               bin_stop = bin_stop)
    h_odd_norm = np.copy(h_E/np.sum(h_E))
    h_even_norm = np.copy(h_G/np.sum(h_G))
    
    if fit: 
    
        G_fit = fit_2D_Gaussian('amp_off_even', bins_G, h_G, 
                                                guessParam[0],
                                                max_fev = 1000,
                                                contour_line = 2)
        E_fit = fit_2D_Gaussian('amp_off_odd', bins_E, h_E,
                                                guessParam[1],
                                                max_fev = 1000,
                                                contour_line = 2)
        F_fit = fit_2D_Gaussian('amp_off_odd', bins_F, h_F,
                                                guessParam[2],
                                                max_fev = 1000,
                                                contour_line = 2)
    
    

        G_fit_h = Gaussian_2D(np.meshgrid(bins_G[:-1], bins_G[:-1]), *G_fit.info_dict['popt'])
        G_fit_h_norm = np.copy(G_fit_h/np.sum(G_fit_h))
        
        E_fit_h = Gaussian_2D(np.meshgrid(bins_E[:-1], bins_E[:-1]), *E_fit.info_dict['popt'])
        E_fit_h_norm = np.copy(E_fit_h/np.sum(E_fit_h))
        
        F_fit_h = Gaussian_2D(np.meshgrid(bins_F[:-1], bins_F[:-1]), *F_fit.info_dict['popt'])
        F_fit_h_norm = np.copy(F_fit_h/np.sum(F_fit_h))
        
        # is_even = hist_discriminant(even_fit_h, odd_fit_h)
        # is_odd = np.logical_not(is_even)
        
        # #debugging
        # # print(np.sum(h_odd), np.sum(h_even))
        # # print(np.sum(h_odd_norm), np.sum(h_even_norm))
        # # print('fid sums', np.sum(h_odd_norm[is_even]), np.sum(h_even_norm[is_odd]))
        
        # plt.pcolormesh(bins_odd, bins_odd, h_odd_norm)
        # plt.colorbar()
        
        # data_fidelity = 1-np.sum(h_odd_norm[is_even], dtype = "float64")-np.sum(h_even_norm[is_odd], dtype = "float64")
        # fit_fidelity = 1-np.sum(odd_fit_h_norm[is_even], dtype = "float64")-np.sum(even_fit_h_norm[is_odd], dtype = "float64")
        
        if plot: 
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(bins_G, bins_G, h_G)
            G_fit.plot_on_ax(ax)
            ax.add_patch(G_fit.sigma_contour())
            ax.set_aspect(1)
            plt.colorbar(pc)
            
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(bins_E, bins_E, h_E)
            E_fit.plot_on_ax(ax)
            ax.add_patch(E_fit.sigma_contour())
            ax.set_aspect(1)
            plt.colorbar(pc)
            
            fig, ax = plt.subplots()
            pc = ax.pcolormesh(bins_F, bins_F, h_F)
            F_fit.plot_on_ax(ax)
            ax.add_patch(F_fit.sigma_contour())
            ax.set_aspect(1)
            plt.colorbar(pc)
            
            # fig, ax = plt.subplots()
            # # pc = ax.pcolormesh(bins_odd, bins_odd, is_even, cmap = 'seismic')
            # amp_off_odd_fit.plot_on_ax(ax)
            # amp_off_even_fit.plot_on_ax(ax)
            # ax.add_patch(amp_off_odd_fit.sigma_contour())
            # ax.add_patch(amp_off_even_fit.sigma_contour())
            # ax.set_aspect(1)
            # plt.colorbar(pc)
    
    
    return None #data_fidelity, fit_fidelity, even_fit, odd_fit
def get_fidelity_vs_records(datapath, plot = False, hist_scale = None, records_per_pulsetype = 3870*2, bin_start = 50, bin_stop = 150): 
    odd_only = 0
    numRecords = records_per_pulsetype
    IQ_offset = (0,0)
    plot = True, 

    I_offset, Q_offset = IQ_offset
    dd = all_datadicts_from_hdf5(datapath)['data']
    
    time_unit = dd['time']['unit']
    
    print(np.size(np.unique(dd['time']['values'])))
    time_vals = dd['time']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    I_plus = dd['I_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    I_minus = dd['I_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    
    Q_plus = dd['Q_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    Q_minus = dd['Q_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    
    print(np.size(I_minus))
    
    #averages
    I_plus_avg = np.average(I_plus, axis = 0)
    I_minus_avg = np.average(I_minus, axis = 0)
    Q_plus_avg = np.average(Q_plus, axis = 0)
    Q_minus_avg = np.average(Q_minus, axis = 0)
    
    if hist_scale == None: 
        hist_scale = 2*np.max(np.array([I_plus_avg, I_minus_avg, Q_plus_avg, Q_minus_avg]))
    
    #re-weave the data back into it's original pre-saved form
    
    bins_even, bins_odd, h_even, h_odd = Process_One_Acquisition(datapath.split('/')[-1].split('\\')[-1], I_plus, I_minus, Q_plus, Q_minus, bin_start, bin_stop, hist_scale = hist_scale, even_only = 0, odd_only = 0, plot = plot)
    
    Plus_x0Guess = np.average(np.average(I_plus_avg[bin_start:bin_stop]))
    Plus_y0Guess = np.average(np.average(Q_plus_avg[bin_start:bin_stop]))
    Plus_ampGuess = np.max(h_even)
    Plus_sxGuess = np.max(bins_even)/5
    Plus_syGuess = Plus_sxGuess
    Plus_thetaGuess = 0
    Plus_offsetGuess = 0
    
    Minus_x0Guess = np.average(np.average(I_minus_avg[bin_start:bin_stop]))
    Minus_y0Guess = np.average(np.average(Q_minus_avg[bin_start:bin_stop]))
    Minus_ampGuess = np.max(h_even)
    Minus_sxGuess = np.max(bins_even)/5
    Minus_syGuess = Minus_sxGuess
    Minus_thetaGuess = 0
    Minus_offsetGuess = 0
    
    guessParams = [[Plus_ampGuess, Plus_x0Guess, Plus_y0Guess, Plus_sxGuess, Plus_syGuess, Plus_thetaGuess],
                   [Minus_ampGuess, Minus_x0Guess, Minus_y0Guess, Minus_sxGuess, Minus_syGuess, Minus_thetaGuess]]