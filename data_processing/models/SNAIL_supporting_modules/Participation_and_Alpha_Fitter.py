# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#bring in actual SNAIL data
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib.colors as color
from matplotlib.widgets import Slider, TextBox
import time
import pickle


#%% Use slider to select modulation curve
def get_fs_data(path): 
    datadict = all_datadicts_from_hdf5(path)['data']
    currents = datadict.extract('phase').data_vals('current')
    freqs = datadict.extract('phase').data_vals('frequency')
    phases = datadict.extract('phase').data_vals('phase')
    mags = datadict.extract('power').data_vals('power')
    return currents, freqs, mags, phases
def convert_to_2D(filt_arr, to_be_2d):
    d2_arr = []
    for val in np.unique(filt_arr):
        d2_arr.append(to_be_2d[filt_arr == val])
    return np.array(d2_arr)
def slider_fit(fs_filepath, fit_filepath, quanta_start, quanta_size, p_arr, alpha_arr):
    currents, freqs, mags, phases = get_fs_data(fs_filepath)
    
    snail_freqs_fits = pickle.load(open(fit_filepath, "rb"))

    #reformat the data into a 2d array by iterating through current values
    phases_2D = convert_to_2D(currents, phases)
    mags_2D = convert_to_2D(currents, mags)
    ind_vars, dep_vars = [np.unique(currents), np.unique(freqs)], [mags_2D, phases_2D]
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.3, bottom=0.4)
    startpoint = quanta_start
    stoppoint = startpoint+quanta_size
    trimleft = np.where(np.unique(currents)>startpoint)[0][0]
    trimright = np.where(np.unique(currents)<stoppoint)[0][-1]
    
    dep_var_trimmed = dep_vars[1][trimleft:trimright]
    ind_vars_trimmed = np.copy(ind_vars)
    ind_vars_trimmed[0] = ind_vars[0][trimleft:trimright]
    colors = [color.hex2color('#0000FF'), color.hex2color('#FFFFFF'), color.hex2color('#FF0000')]
    _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
    adj = 0
    graph = dep_var_trimmed.T
    low = np.min(dep_var_trimmed)
    high = np.max(dep_var_trimmed)
    
    _norm = color.Normalize(vmin = low, vmax = high)
    
    scale_factor = np.pi*2/(np.max(ind_vars_trimmed[0])-np.min(ind_vars_trimmed[0]))
    graph_x = ind_vars_trimmed[0]*scale_factor
    
    dplot = plt.imshow(graph, alpha = 0.5, extent = [0, 2*np.pi, ind_vars_trimmed[1][0], ind_vars_trimmed[1][-1]], aspect = 'auto', origin = 'lower', norm = _norm, cmap = _cmap)
    fplot, = plt.plot(np.linspace(0,2*np.pi, 51), snail_freqs_fits[0][0])
    ax.margins(x=0)
    
    def update(val):
        p = int(sp.val)
        alpha = int(salpha.val)
        start_freq = sfreq.val
        # x = np.where(p_arr == p)[0][0]
        # y = np.where(alpha_arr == alpha)[0][0]
        # print("p: "+str(p_arr[p])+" Alpha: "+str(alpha_arr[alpha]))
        scale_factor = start_freq/snail_freqs_fits[p][alpha][0]
        fplot.set_ydata(snail_freqs_fits[p][alpha]*scale_factor)
        
        fig.canvas.draw_idle()
        
    def submit(text):
        center = float(text)
        x_adj = center-np.pi
        fplot.set_xdata(phi+x_adj)
        fig.canvas.draw_idle()
    
    axp = plt.axes([0.25, 0.1, 0.65, 0.03])
    axalpha = plt.axes([0.25, 0.15, 0.65, 0.03])
    axbox = plt.axes([0.25, 0.2, 0.8, 0.075])
    axfreq = plt.axes([0.25, 0.05,0.65, 0.03])
    
    text_box = TextBox(axbox, 'Center', initial="3.1415")
    text_box.on_submit(submit)
    
    sp = Slider(axp, 'P index',0, np.size(p_arr)-1, valinit=0, valstep=1)
    salpha = Slider(axalpha, 'Alpha index', 0, np.size(alpha_arr)-1, valinit=0, valstep=1)
    sfreq = Slider(axfreq, 'start frequency', 6e9, 6.5e9, valinit=6e9)
    
    sp.on_changed(update)
    salpha.on_changed(update)
    sfreq.on_changed(update)
    
    return sp, salpha, sfreq

if __name__ == '__main__': 
    fs_filepath= r"\\136.142.53.51\data002\Texas\Cooldown_20210408\SA_C1_FS\2021-05-04_0005_C1_FS6_very_wide_fine.ddh5"
    # load pickled fit data
    fit_filepath = r"C:\Users\Hatlab_3\Desktop\RK_Scripts\NewRepos\data_processing\data_processing\models\SNAIL_supporting_modules\SNAIL_detailed.p"
    p_arr = np.linspace(0.01, 0.3, 50)
    alpha_arr = np.linspace(0.1, 0.32, 50)
    p_slider, a_slider, freq_slider = slider_fit(fs_filepath, fit_filepath, -8.98e-5, 220.6e-6, p_arr, alpha_arr)