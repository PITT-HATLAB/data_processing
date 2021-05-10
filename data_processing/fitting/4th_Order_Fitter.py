# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:39:22 2020

@author: Tzu-Chiao Chien and Ryan Kaufman
"""


"""
Created on Mon Jul 24 19:45:51 2017

@author: Ryan Kaufman

Will eventually be a module for grabbing data out of files via a GUI
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as color
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import easygui

cwd = easygui.diropenbox('Select where you are working')
print(cwd)
u#%%
#Single File, single traces overlaid onto one-another if you choose: 
def get_plot(file = None):
    if file == None: 
        filepath = easygui.fileopenbox("Choose File", default = cwd)
        file = h5py.File(filepath)
        assert file != None
    datasets = file.keys()
    ind_var = easygui.choicebox("Pick the independent variable: ", choices = datasets)
    print(ind_var)
    dep_var = easygui.choicebox("Pick the dependent variable: ", choices = datasets)
    print(dep_var)
    
    ind_var_data = np.array(file[ind_var])
    dep_var_data = np.array(file[dep_var])
    
    
    return ind_var_data, dep_var_data, ind_var, dep_var

ind_vars, dep_vars, ind_name, dep_name = get_plot()
# ind_vars2, dep_vars2, ind_name2, dep_name2 = get_plot()

#%%
print("Max: {:.3f} \nMin: {:.3f}".format(np.max(dep_vars), np.min(dep_vars)))
plt.close("all")
plt.plot(ind_vars, dep_vars, label = "Forward")
# plt.plot(ind_vars2[0], dep_vars2[0], label = "Backward")
plt.legend()
plt.xlabel(ind_name)
plt.ylabel(dep_name)
plt.title(ind_name +' vs.'+dep_name)

#%%
# #%% Get the data from multiple separate files
def compile_traces():
    num_traces = easygui.choicebox(choices = np.arange(1,6))
    i = 0 
    trace_data = []
    freq_data = []
    while i < num_traces: 
        filepath = easygui.fileopenbox("Choose File #"+str(i+1), default = cwd)
        file = h5py.File(filepath)
        trace = np.array(file['noise']) #TODO: generalize to choice
        freqs = np.array(file['Freq'])
        trace_data.append(trace)
        freq_data.append(freqs)
        file.close()
        i+=1
    return freq_data, trace_data

#%% Extract 2d Data from h5
def get_pcolor(cwd): 
    filepath = easygui.fileopenbox("Choose File", default = cwd)
    file = h5py.File(filepath)
    datasets = file.keys()
    ind_vars = easygui.multchoicebox("Pick the independent variables: ", choices = datasets)
    print(ind_vars)
    dep_var_name = easygui.choicebox("Pick the dependent variable: ", choices = datasets)
    print(dep_var_name)
    #checking for data redundancy, i.e. if every line of an independent variable is the same, reduce it to just the first line
    ind_var_data = []
    for ind_var in ind_vars: 
        ind_var_datum = np.array(file[ind_var])
        is_repetitive = True
        for i in range(np.shape(ind_var_datum)[0]):
            if np.any(ind_var_datum[i] != ind_var_datum [0]): 
                is_repetitive = False
        if is_repetitive == True: 
            ind_var_datum = ind_var_datum[0]
        ind_var_data.append(ind_var_datum)
            
    dep_var_data = np.array(file[dep_var_name])
    
    return ind_var_data, ind_vars, dep_var_data, dep_var_name
    
ind_vars,ind_var_names, dep_var, dep_var_name = get_pcolor(cwd)
# ind_vars2,ind_var_names2, dep_var2, dep_var_name2 = get_pcolor(cwd)

#%%Plot Pcolor
#TODO: incorporate into get_pcolor with more color choices, etc
ind_avgs = [np.average(i) for i in ind_vars]
dep_avg = np.average(dep_var)
colors = [color.hex2color('#0000FF'), color.hex2color('#FFFFFF'), color.hex2color('#FF0000')]        
_cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
adj = 5
_norm = color.Normalize(vmin = np.min(dep_var)+adj, vmax = np.max(dep_var)-adj)
# x = np.min(dep_var)
# y = np.max(dep_var)
plt.pcolormesh((ind_vars[1])/1e6,ind_vars[0], dep_var, cmap = _cmap, norm = _norm)
plt.colorbar(label = 'S21 Phase (Deg)')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.xlabel( 'VNA_Frequency (MHz)')
plt.ylabel('VNA Input Power (dBm)')
plt.title('S21 Phase vs. Frequency and Input Power')
#%% Now extract a line cut from that 2d plot

#from the plot, we're going to want to be able to specify one value of one of 
#the independent variables, then the line cuts across the other one at that 
#value

#ideally this would be a slider in the plot window, but whatever
def linecut_data(ind_vars, ind_var_names, dep_var, dep_var_name):
    cut_var_name = easygui.choicebox("Choose variable you would like to cut along:", choices = ind_var_names)
    cut_name_index = list(ind_var_names).index(cut_var_name) #only lists have the .index method :(
    #Sometimes the order of names does not correspond to the ordering of indices in dep_var, this takes care of that
    cut_index = np.shape(dep_var).index(len(ind_vars[cut_name_index]))
    cut_var_val = float(easygui.choicebox("Choose value of the cut variable:", choices = ind_vars[cut_name_index]))
    cut_var_val_index = list(ind_vars[cut_name_index]).index(cut_var_val)
    #we know it is 2d data, so if cut_index is 0, we want a [cut_var_index,:] cut, else [:,cut_var_index]
    print(dep_var_name)
    #TODO: find more pythonic solution
    if cut_index == 0: 
        cut_dep_data = dep_var[cut_var_val_index, :]
        
    elif cut_index == 1: 
        cut_dep_data = dep_var[:, cut_var_val_index]

    cut_ind_data = ind_vars[int(cut_index)] #this logic just makes 0's into 1's and vice-versa
    cut_ind_name = ind_var_names[int(cut_index)]
    cut_dep_name = "{} at {} = {:.3f}".format(dep_var_name, cut_var_name, cut_var_val)
    
    return cut_ind_data, cut_ind_name, cut_dep_data, cut_dep_name
    
cut_ind_data, cut_ind_name, cut_dep_data, cut_dep_name = linecut_data(ind_vars,ind_var_names, dep_var, dep_var_name)

#%% Plot the linecut

plt.plot(cut_ind_data, cut_dep_data, '.')
plt.title(cut_dep_name)
plt.xlabel(cut_ind_name)
plt.ylabel("")

#%% Finding derivatives

def get_min_ders(freqs, data):
    mags = np.array(data)
    ders = []
    der_max_locs = [] #should correspond to bifurcation
    der_max_freqs = []
    for trace in mags: 
        der = np.gradient(trace)
        der_max_loc = list(np.where(der == np.min(der))[0])[0]
        der_max_locs.append(der_max_loc)
        der_max_freq = freqs[der_max_loc]
        der_max_freqs.append(der_max_freq)
        ders.append(der)
    return(der_max_freqs)

freqs_of_max_change = get_min_ders(ind_vars[1],dep_var)
linearized_voltages = np.power(10,ind_vars[0]/20)
offset_from_right = -2
fitx = linearized_voltages[0:offset_from_right-1]
fity = (freqs_of_max_change[0:offset_from_right-1]-freqs_of_max_change[0])/1e6

#Fit Function to polynomial
#%% Numpy Fit
from numpy.polynomial.polynomial import polyfit
fit = np.flip(np.polyfit(fitx,fity,4))
fit_func = lambda x,a,b,c,d,e: a+b*x+c*x**2+d*x**3+e*x**4
english_sucks = ["th","st","nd","rd","th"]
for coefficient in enumerate(fit):
    print("{}{} order term: {}".format(coefficient[0],english_sucks[coefficient[0]],coefficient[1]))
plt.plot(fitx,fity,'o')
plt.plot(fitx,fit_func(fitx, *fit))
plt.title("All Order Fit")
#%% Scipy Fit
from scipy.optimize import curve_fit
def fit_func(x,a,b): 
    return a*x+b*x**2
fit_end, other_stuff = curve_fit(fit_func, fitx,fity)
print("First order: {} (MHz/mV)\nSecond order: {} (MHz/mV)^2".format(fit_end[0],fit_end[1]))
plt.plot(fitx,fity,'o')
plt.plot(fitx,fit_func(fitx,fit_end[0],fit_end[1]))
plt.xlabel("Linearized Voltage ($\sqrt{mW}$)")
plt.ylabel("Detuning (MHz)")
plt.title("1st and 2nd order fit")
#%%












