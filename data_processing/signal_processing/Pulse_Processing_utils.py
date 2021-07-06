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

import numpy as np
import matplotlib.pyplot as plt

from plottr.data.datadict_storage import all_datadicts_from_hdf5

def Process_One_Acquisition(name, sI_c1, sI_c2, sQ_c1 ,sQ_c2, bin_start, bin_stop, hist_scale = 200, odd_only = False, even_only = False, plot = False):
    if plot: 
        fig = plt.figure(1, figsize = (12,8))
        fig.suptitle(name)
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
    
    
    
def boxcar_histogram(fig, ax, start_pt, stop_pt, sI, sQ, Ioffset = 0, Qoffset = 0, scale = 1, num_bins = 100):
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

from scipy.optimize import curve_fit


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
                    max_fev = 100, 
                    contour_line = 3): 
    
    X, Y = np.meshgrid(bins[0:-1], bins[0:-1])
    resh_size = np.shape(X)
    xdata, ydata= np.vstack((X.ravel(), Y.ravel())), h_arr.ravel()
    # print('xdata_shape: ', np.shape(xdata))
    # print("y shape: ",np.shape(ydata))
    print("running curve_fit")
    #,amplitude, xo, yo, sigma_x, sigma_y, theta
    bounds = [[0,np.min(bins), np.min(bins), 0, 0, 0],
              [np.max(h_arr), np.max(bins), np.max(bins), np.max(bins), np.max(bins), 2*np.pi]]
    popt, pcov = curve_fit(Gaussian_2D, xdata, ydata, p0 = guessParams, maxfev = max_fev)
    GC = Gaussian_info()
    GC.info_dict['name'] = name
    GC.info_dict['canvas'] = xdata 
    GC.info_dict['amplitude'] = popt[0]
    GC.info_dict['x0'] = popt[1]
    GC.info_dict['y0'] = popt[2]
    GC.info_dict['sigma_x'] = np.abs(popt[3])
    GC.info_dict['sigma_y'] = np.abs(popt[4])
    GC.info_dict['theta'] = popt[5]
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
    time_vals = dd['time']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    rec_unit = dd['record_num']['unit']
    rec_num = dd['record_num']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))
    
    I_plus = dd['I_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    I_minus = dd['I_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-I_offset
    
    Q_plus = dd['Q_plus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    Q_minus = dd['Q_minus']['values'].reshape((numRecords//2, np.size(dd['time']['values'])//(numRecords//2)))-Q_offset
    
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
        
    
    