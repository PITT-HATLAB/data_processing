# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:17:43 2021

@author: Hatlab_3
"""
import numpy as np
import time
from scipy.signal import butter, sosfilt, sosfreqz
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Normalize as Norm

#DO NOT USE, only here for backwards compatibility
def demod(signal_data, reference_data, mod_freq = 50e6, sampling_rate = 1e9): 
    '''
    #TODO: 
    Parameters
    ----------
    signal_data : np 1D array - float64
        signal datapoints
    reference_data : np 1D array - float64
        reference datapoints
    mod_freq : float64, optional
        Modulation frequency in Hz. The default is 50e6.
    sampling_rate : float64, optional
        sampling rate in samples per second. The default is 1e9.

    Returns
    -------
    sig_I_summed : np 1D array - float64
        Signal multiplied by Sine and integrated over each period.
    sig_Q_summed : np 1D array - float64
        Signal multiplied by Cosine and integrated over each period.
    ref_I_summed : np 1D array - float64
        reference multiplied by sine and integrated over each period.
    ref_Q_summed : np 1D array - float64
        Reference multiplied by Cosine and integrated over each period.
    '''
    
    '''first pad the arrays to get a multiple of the number of samples in a 
    demodulation period, this will make the last record technically inaccurate 
    but there are thousands being 
    averaged so who cares
    '''
    
    #first demodulate both channels
    # print("Signal Data Shape: ",np.shape(signal_data))
    # print("Reference Data Shape: ",np.shape(reference_data))
    period = int(sampling_rate/mod_freq)
    print("Integrating over a period of: ", period)
    signal_data = np.pad(signal_data, (0,int(period-np.size(signal_data)%period)))
    reference_data = np.pad(reference_data, (0,int(period-np.size(reference_data)%period)))
    # print("Signal Data Shape: ",np.shape(signal_data))
    # print("Reference Data Shape: ",np.shape(reference_data))
    point_number = np.arange(np.size(reference_data))
    # print('Modulation period: ', period)
    SinArray = np.sin(2*np.pi/period*point_number)
    CosArray = np.cos(2*np.pi/period*point_number)
    
    sig_I = signal_data*SinArray
    sig_Q = signal_data*CosArray
    ref_I = reference_data*SinArray
    ref_Q = reference_data*CosArray
    
    #now you cut the array up into periods of the sin and cosine modulation, then sum within one period
    #the sqrt 2 is the RMS value of sin and cosine squared, period is to get rid of units of time
    
    sig_I_summed = np.sum(sig_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    sig_Q_summed = np.sum(sig_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_I_summed = np.sum(ref_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_Q_summed = np.sum(ref_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    
    return (sig_I_summed, sig_Q_summed, ref_I_summed, ref_Q_summed)

def demod_period(signal_data, reference_data, period = 20, sampling_rate = 1e9, debug = False): 
    '''
    #TODO: 
    Parameters
    ----------
    signal_data : np 1D array - float64
        signal datapoints
    reference_data : np 1D array - float64
        reference datapoints
    mod_freq : float64, optional
        Modulation frequency in Hz. The default is 50e6.
    sampling_rate : float64, optional
        sampling rate in samples per second. The default is 1e9.

    Returns
    -------
    sig_I_summed : np 1D array - float64
        Signal multiplied by Sine and integrated over each period.
    sig_Q_summed : np 1D array - float64
        Signal multiplied by Cosine and integrated over each period.
    ref_I_summed : np 1D array - float64
        reference multiplied by sine and integrated over each period.
    ref_Q_summed : np 1D array - float64
        Reference multiplied by Cosine and integrated over each period.
    '''
    
    '''first pad the arrays to get a multiple of the number of samples in a 
    demodulation period, this will make the last record technically inaccurate 
    but there are thousands being 
    averaged so who cares
    '''
    
    #first demodulate both channels
    # print("Signal Data Shape: ",np.shape(signal_data))
    # print("Reference Data Shape: ",np.shape(reference_data))
    if debug: 
        print("Integrating over a period of: ", period)
        print("Implies a demodulation frequency of: ", sampling_rate/period/1e6, "MHz")
    signal_data = np.pad(signal_data, (0,int(period-np.size(signal_data)%period)))
    reference_data = np.pad(reference_data, (0,int(period-np.size(reference_data)%period)))
    # print("Signal Data Shape: ",np.shape(signal_data))
    # print("Reference Data Shape: ",np.shape(reference_data))
    point_number = np.arange(np.size(reference_data))
    # print('Modulation period: ', period)
    SinArray = np.sin(2*np.pi/period*point_number)
    CosArray = np.cos(2*np.pi/period*point_number)
    
    sig_I = signal_data*SinArray
    sig_Q = signal_data*CosArray
    ref_I = reference_data*SinArray
    ref_Q = reference_data*CosArray
    
    #now you cut the array up into periods of the sin and cosine modulation, then sum within one period
    #the sqrt 2 is the RMS value of sin and cosine squared, period is to get rid of units of time
    
    sig_I_summed = np.sum(sig_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    sig_Q_summed = np.sum(sig_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_I_summed = np.sum(ref_I.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    ref_Q_summed = np.sum(ref_Q.reshape(np.size(sig_I)//period, period), axis = 1)*(np.sqrt(2)/period)
    
    return (sig_I_summed, sig_Q_summed, ref_I_summed, ref_Q_summed)

def demod_all_records(s_array: np.ndarray, r_array: np.ndarray, period = 20):
    '''
    Parameters
    ----------
    s_array : np.ndarray
        array of signal data [R, t] where R is records, t is time
    r_array : np.ndarray
        array of reference data [R, t] where R is records, t is time
    period : int, optional
        window over which to integrate for demodulation, with 1GS/s the unit is ns. The default is 20ns == 50MHz.

    Returns
    -------
    s_demod_arr : TYPE
        array of demodulated 
    r_demod_arr : TYPE
        DESCRIPTION.

    '''
    sI_arr = []
    sQ_arr = []
    rI_arr = []
    rQ_arr = []
    
    #demodulate each record in windows of (period) width
    for rec_sig, rec_ref in zip(s_array, r_array): 
        sI, sQ, rI, rQ = demod_period(rec_sig, rec_ref, period = period)
        sI_arr.append(sI)
        sQ_arr.append(sQ)
        rI_arr.append(rI)
        rQ_arr.append(rQ)
    
    #turn everything into numpy arrays
    for data in [sI_arr, sQ_arr, rI_arr, rQ_arr]: 
        data = np.array(data)
    
    return [sI_arr, sQ_arr], [rI_arr, rQ_arr]


bpf_func = lambda cfreq, BW, order: butter(order, [cfreq-BW/2, cfreq+BW/2], fs = 1e9, output = 'sos', btype = 'bandpass')

def filter_all_records(s_array, r_array, filt): 
    s_filt_arr = []
    r_filt_arr = []
    
    #demodulate each record in windows of (period) width
    for rec_sig, rec_ref in zip(s_array, r_array): 
        s_filt = sosfilt(filt, rec_sig)
        r_filt = sosfilt(filt, rec_ref)
        s_filt_arr.append(s_filt)
        r_filt_arr.append(r_filt)
        
    #turn everything into numpy arrays
    for data in [s_filt_arr, r_filt_arr]: 
        data = np.array(data)
    
    return s_filt_arr, r_filt_arr
    
    
def remove_offset(I, Q, window = [0, 40]): 
    #remove an IQ offset from the data
    offset_data = I-np.average(I[:, window[0]: window[1]]), Q-np.average(Q[:, window[0]: window[1]])
    return offset_data
    
    
def phase_correction(sigI, sigQ, refI, refQ): 
    '''
    Parameters
    ----------
    sigI : np 2D array - (records,samples) float64
        demodulated signal - In-phase
    sigQ : np 2D array - (records,samples) float64
        demodulated signal - Quadrature phase
    refI : np 2D array - (records,samples) float64
        demodulated reference - In-phase
    refQ : np 2D array - (records,samples) float64
        demodulated reference - quadrature-phase
        
    Note: reference and signal arrays must all be of the same length

    Returns
    -------
    sigI_corrected : np 2D array - float64
        Signal I rotated by reference phase averaged over each record
    sigQ_corrected : np 2D array - float64
        Signal Q rotated by reference phase averaged over each record

    '''
    sigI_corrected = np.zeros(np.shape(sigI))
    sigQ_corrected = np.zeros(np.shape(sigQ))
    rI_trace = np.zeros(np.shape(sigI)[0])
    rQ_trace =np.zeros(np.shape(sigI)[0])
    for i, (sI_rec, sQ_rec, rI_rec, rQ_rec) in enumerate(zip(sigI,sigQ,refI,refQ)):
        
        rI_avg, rQ_avg = np.average(rI_rec), np.average(rQ_rec)
        
        rI_trace[i], rQ_trace[i] = rI_avg, rQ_avg
        
        Ref_mag = np.sum(np.sqrt(rI_avg**2 + rQ_avg**2))
        
        sigI_corrected[i] = (sI_rec*rI_avg + sQ_rec*rQ_avg)/Ref_mag
        sigQ_corrected[i] = (-sI_rec*rQ_avg + sQ_rec*rI_avg)/Ref_mag
    
    return sigI_corrected, sigQ_corrected, rI_trace, rQ_trace

def generate_matched_weight_funcs(data1, data2, bc = False, bc_window = [50, 150]):
    
    d1I, d1Q = data1
    d2I, d2Q = data2
    
    if bc == False: 
    
        WF_I = np.average(d1I, axis = 0)-np.average(d2I, axis = 0)
        WF_Q = np.average(d1Q, axis = 0)-np.average(d2Q, axis = 0)
    else: 
        WF_I = np.zeros(np.shape(d1I)[1])
        WF_Q = np.zeros(np.shape(d1I)[1])
        
        WF_I[bc_window[0]: bc_window[1]] = 1/(bc_window[1]-bc_window[0])
        WF_Q[bc_window[0]: bc_window[1]] = 1/(bc_window[1]-bc_window[0])
        
    return WF_I, WF_Q
    
def weighted_histogram(weight_function_arr_I, weight_function_arr_Q, sI, sQ, scale = 1, num_bins = 100, record_track = False, plot = False, fig = None, ax = None): 
    I_pts = []
    Q_pts = []

    for I_row, Q_row in zip(sI, sQ): 
        I_pts.append(np.dot(I_row, weight_function_arr_I)+np.dot(Q_row, weight_function_arr_Q))
        Q_pts.append(np.dot(Q_row, weight_function_arr_I)-np.dot(I_row, weight_function_arr_Q))

    bins = np.linspace(-1,1, num_bins)*scale
    (h, xedges, yedges) = np.histogram2d(I_pts, Q_pts, bins = [bins, bins], density = False)
    
    if plot: 
        im = ax.pcolormesh(bins, bins, h)
        divider = make_axes_locatable(ax)
        ax.set_aspect(1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax = cax, orientation = 'vertical')
    
    if record_track: 
        fig2, ax2 = plt.subplots()
        ax2.set_title("Record Tracking: Demodulated signals")
        ax2.set_xlabel("time (~us)")
        ax2.set_ylabel("$\phi(t)$")
        unwrapped_phases = np.mod(np.unwrap(np.arctan(np.array(sI[0:500, 100])/np.array(sQ[0:500, 100])), period = np.pi), 2*np.pi)
        ax2.plot(np.arange(100)*500, unwrapped_phases, '.', label = "phi(t)")
        print("Average phase difference between records: ", np.average(np.diff(unwrapped_phases))/np.pi*180, ' degrees')

    return bins, h, I_pts, Q_pts

def Gaussian_2D(M,amplitude, xo, yo, sigma):
    theta = 0
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
        ax.annotate("", xy=self.center_vec(), xytext=(0,0), arrowprops=dict(arrowstyle = '->', lw = 3, color = color))
    def plot_array(self):
        return Gaussian_2D(*self.info_dict['popt'])
    def sigma_contour(self): 
        x0, y0 = self.center_vec()
        sx = self.info_dict['sigma_x']
        sy = self.info_dict['sigma_y']
        # angle = self.info_dict['theta']
        angle = 0
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
                    contour_line = 0, 
                    debug = False): 
    if debug: 
        print("fitting with maxfev = ", max_fev)
    X, Y = np.meshgrid(bins[0:-1], bins[0:-1])
    xdata, ydata= np.vstack((X.ravel(), Y.ravel())), h_arr.ravel()
    # print('xdata_shape: ', np.shape(xdata))
    # print("y shape: ",np.shape(ydata))
    #,amplitude, xo, yo, sigma_x, sigma_y, theta
    bounds = ([0,np.min(bins), np.min(bins), 0],
              [10*np.max(h_arr), np.max(bins), np.max(bins), np.max(bins)])
    # print(bounds)
    popt, pcov = curve_fit(Gaussian_2D, xdata, ydata, p0 = guessParams, maxfev = max_fev, bounds = bounds)
    GC = Gaussian_info()
    GC.info_dict['name'] = name
    GC.info_dict['canvas'] = xdata 
    GC.info_dict['amplitude'] = popt[0]
    GC.info_dict['x0'] = popt[1]
    GC.info_dict['y0'] = popt[2]
    GC.info_dict['sigma_x'] = np.abs(popt[3])
    GC.info_dict['sigma_y'] = np.abs(popt[3])
    # GC.info_dict['theta'] = popt[4]
    GC.info_dict['popt'] = popt
    GC.info_dict['pcov'] = pcov
    # GC.info_dict['contour'] = get_contour_line(X, Y, Gaussian_2D(xdata, *popt).reshape(resh_size), contour_line = contour_line)
    
    return GC

def hist_discriminant(h1, h2):
    #1 if in h1, 0 if in h2
    return ((h1-h2)>0)

def majorityVote3State(all_I, all_Q, WFS, vote_maps, bins, plot = 1, debug = 0, num_bins = 100): 
    [GE_is_G, GF_is_G, EF_is_E] = vote_maps #purely expanatory
    GE_is_E = np.logical_not(GE_is_G)
    GF_is_F = np.logical_not(GF_is_G)
    EF_is_F = np.logical_not(EF_is_E)
    
    [Sge_I, Sge_Q], [Sgf_I, Sgf_Q], [Sef_I, Sef_Q] = WFS
    
    results = []
    GE_results = []
    GF_results = []
    EF_results = []
    
    for i, record in enumerate(list(zip(all_I, all_Q))): 
        
        It, Qt = record[0], record[1]
        
        #GE weights
        ge_I = np.dot(Sge_I, It)+np.dot(Sge_Q, Qt)
        # print('record ', i)
        # print('shape', np.shape(all_I))
        
        ge_Q = np.dot(Sge_I, Qt)-np.dot(Sge_Q, It)
        
        Iloc = np.digitize(ge_I, bins)
        # print("Iloc", Iloc)
        Qloc = np.digitize(ge_Q, bins)
        
        if Iloc >= num_bins-1: Iloc = num_bins-2
        if Qloc >= num_bins-1: Qloc = num_bins-2
        
        #if 1 it's G
        Sge_result = GE_is_G[Iloc, Qloc]
        
        #GF weights
        gf_I = np.dot(Sgf_I, It)+np.dot(Sgf_Q, Qt)
        gf_Q = np.dot(Sgf_I, Qt)-np.dot(Sgf_Q, It)
        
        Iloc = np.digitize(gf_I, bins)
        Qloc = np.digitize(gf_Q, bins)
        
        if Iloc >= num_bins-1: Iloc = num_bins-2
        if Qloc >= num_bins-1: Qloc = num_bins-2
        
        #if 1 it's G
        Sgf_result = GF_is_G[Iloc, Qloc]
        
        #EF weights
        ef_I = np.dot(Sef_I, It)+np.dot(Sef_Q, Qt)
        ef_Q  = np.dot(Sef_I, Qt)-np.dot(Sef_Q, It)
        
        Iloc = np.digitize(ef_I, bins)
        Qloc = np.digitize(ef_Q, bins)#edge-shifting
        
        if Iloc >= num_bins-1: Iloc = num_bins-2
        if Qloc >= num_bins-1: Qloc = num_bins-2
        
        #if 1 it's E
        Sef_result = EF_is_E[Iloc, Qloc]
        
        
        # print(Sge_result)
        # print(Sgf_result)
        if Sge_result*Sgf_result: 
            result = 1 #G
        elif not Sge_result and Sef_result: 
            result = 2 #E
        elif not Sef_result and not Sgf_result: 
            result = 3 #F
        else: 
            result = 4 #Null
        
        results.append(result)
        GE_results.append(Sge_result)
        GF_results.append(Sgf_result)
        EF_results.append(Sef_result)
        
    results = np.array(results)
    
    #rescale so G-> 1, E-> 2, F -> 3
    GE_results = np.logical_not(np.array(GE_results))+1
    GF_results = np.logical_not(np.array(GF_results))*2+1
    EF_results = np.logical_not(np.array(EF_results))+2
    div1 = np.shape(all_I)[0]//3
    numRecords = 3*div1
    # print(div1)
    correct_classifications = np.append(np.append(np.ones(div1), 2*np.ones(div1)), 3*np.ones(div1))
    
    numberNull = np.sum(results[results == 4]/4)
    fidelity = np.round(np.sum(correct_classifications==results)/numRecords, 3)
    
    if plot: 
        fig, ax = plt.subplots(5,1, figsize = (4, 8))
    
        viridisBig = cm.get_cmap('viridis', 512)
        _cmap = ListedColormap(viridisBig(np.linspace(0, 1, 256)))
        
        scale = Norm(vmin = 1, vmax = 4)
        
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
        # ax[2].imshow([right, right], interpolation = 'none')
        # ax[2].set_aspect(1000)
        fig.tight_layout(h_pad = 1, w_pad = 1)

    # if debug: 
    #     print("checking sum: ", np.max(correct_classifications[2*div1:-1]==results[2*div1:-1]))
    #     print("Number of Null results: ", numberNull)
    #     print("Sge Imbar/sigma: ", np.linalg.norm(GE_G_fit.center_vec()-GE_E_fit.center_vec())/GE_G_fit.info_dict['sigma_x'])
    #     print("Sgf Imbar/sigma: ", np.linalg.norm(GF_G_fit.center_vec()-GF_F_fit.center_vec())/GF_G_fit.info_dict['sigma_x'])
    #     print("Sef Imbar/sigma: ", np.linalg.norm(EF_E_fit.center_vec()-EF_F_fit.center_vec())/EF_E_fit.info_dict['sigma_x'])
    
    G_fidelity = np.round(np.sum(correct_classifications[0:div1]==results[0:div1])/div1, 3)
    E_fidelity = np.round(np.sum(correct_classifications[div1:2*div1]==results[div1:2*div1])/div1, 3)
    F_fidelity = np.round(np.sum(correct_classifications[2*div1:-1]==results[2*div1:-1])/div1, 3)
    
    return G_fidelity, E_fidelity, F_fidelity, numberNull