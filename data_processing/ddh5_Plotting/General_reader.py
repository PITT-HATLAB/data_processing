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

if __name__ == '__main__':
    
    #%%
    cwd = r'H:\Data\Fridge Texas\General measurement\20180323_cable_measurement_after_upgrade'
    if cwd == None: 
        raise Exception("CWD not chosen!")
        
    #%%
    #Single File, single traces overlaid onto one-another if you choose: 
    def get_plot(file = None, IQ = False):
        if file == None: 
            filepath = easygui.fileopenbox("Choose File", default = cwd)
            file = h5py.File(filepath)
            assert file != None
        else: 
            filepath = file
            file = h5py.File(filepath)
            
        datasets = file.keys()
        ind_var = easygui.choicebox("Pick the independent variable: ", choices = datasets)
        # print(ind_var)
        dep_vars = easygui.multchoicebox("Pick the dependent variable: ", choices = datasets)
        # print(dep_var)
        
        ind_var_data = np.array(file[ind_var])
        dep_var_data = []
        for dep_var in dep_vars: 
            dep_var_data.append(np.array(file[dep_var]))
        
        
        return ind_var_data, dep_var_data, ind_var, dep_vars
    #%%
    ind_vars, dep_vars, ind_name, dep_names = get_plot()
    # ind_vars2, dep_vars2, ind_name2, dep_name2 = get_plot()
    #%%
    print(dep_names)
    print(dep_vars)
    i = 0
    for (dep_var, dep_name) in zip(dep_vars,dep_names): 
        print(i)
        plt.figure(i)
        i+=1
        print("Max: {:.3f} \nMin: {:.3f}".format(np.max(dep_var), np.min(dep_var)))
        for i in range(np.shape(np.shape(dep_var))[2]): 
            plt.plot(ind_vars, dep_var)
            plt.xlabel(ind_name)
            plt.ylabel(dep_name)                                                    
            plt.title(f"{i}")
    
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
        dep_vars = easygui.multchoicebox("Pick the dependent variables: ", choices = datasets)
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
        dep_var_data = []
        for dep_var in dep_vars: 
            dep_var_data.append(np.array(file[dep_var]))
        
        return ind_var_data, ind_vars, dep_var_data, dep_vars
        
    ind_vars,ind_var_names, dep_vars, dep_var_names = get_pcolor(cwd)
    # ind_vars2,ind_var_names2, dep_var2, dep_var_name2 = get_pcolor(cwd)
    #%%Plot Pcolor(s)
    #TODO: incorporate into get_pcolor with more color choices, etc
    i = 0
    for (dep_var, dep_var_name) in zip(dep_vars, dep_var_names):
        # print(dep_var)
        plt.figure(i)
        i+=1
        ind_avgs = [np.average(i) for i in ind_vars]
        dep_avg = np.average(dep_var)
        colors = [color.hex2color('#0000FF'), color.hex2color('#FFFFFF'), color.hex2color('#FF0000')]        
        _cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
        adj = 0
        graph = dep_var.T
        low = np.min(dep_var)
        high = np.max(dep_var)
        # low = -13
        # high = 6
        _norm = color.Normalize(vmin = low, vmax = high)
        # x = np.min(dep_var)
        # y = np.max(dep_var)
        plt.pcolormesh((ind_vars[0]),ind_vars[1],graph, cmap = _cmap, norm = _norm)
        plt.colorbar(label = dep_var_name)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel( ind_var_names[0])
        plt.ylabel(ind_var_names[1])
        # plt.title('SHARC33 A_to_S_leakage')
    #%% Now extract a line cut from that 2d plot
    
    #from the plot, we're going to want to be able to specify one value of one of 
    #the independent variables, then the line cuts across the other one at that 
    #value
    
    #ideally this would be a slider in the plot window, but whatever
    def linecut_data(ind_vars, ind_var_names, dep_vars, dep_var_names):
        dep_var_name = easygui.choicebox("Choose dependent variable of cut", choices = dep_var_names)
        dep_var = dep_vars[dep_var_names.index(dep_var_name)]
        cut_var_name = easygui.choicebox("Choose variable you would like to cut along:", choices = ind_var_names)
        cut_name_index = list(ind_var_names).index(cut_var_name) #only lists have the .index method :(
        #Sometimes the order of names does not correspond to the ordering of indices in dep_var, this takes care of that
        cut_index = np.shape(dep_var).index(len(ind_vars[cut_name_index]))
        cut_var_val = float(easygui.choicebox("Choose value of the cut variable:", choices = ind_vars[cut_name_index]))
        cut_var_val_index = list(ind_vars[cut_name_index]).index(cut_var_val)
        #we know it is 2d data, so if cut_index is 0, we want a [cut_var_index,:] cut, else [:,cut_var_index]
        #TODO: find more pythonic solution
        if cut_index == 0: 
            cut_dep_data = dep_var[cut_var_val_index, :]
            
        elif cut_index == 1: 
            cut_dep_data = dep_var[:, cut_var_val_index]
    
        cut_ind_data = ind_vars[int(not cut_index)] #this logic just makes 0's into 1's and vice-versa
        cut_ind_name = ind_var_names[int(not cut_index)]
        cut_dep_name = "{} at {} = {:.3f}".format(dep_var_name, cut_var_name, cut_var_val)
        
        return cut_ind_data, cut_ind_name, cut_dep_data, cut_dep_name
        
    cut_ind_data, cut_ind_name, cut_dep_data, cut_dep_name = linecut_data(ind_vars,ind_var_names, dep_vars, dep_var_names)
    # cut_ind_data2, cut_ind_name2, cut_dep_data2, cut_dep_name2 = linecut_data(ind_vars,ind_var_names, dep_var, dep_var_name)
    #%% Plot the linecut
    
    plt.plot(cut_ind_data, cut_dep_data, '-', label = 'Forward')
    # plt.plot(cut_ind_data2, cut_dep_data2, '-', label = 'Backward')
    plt.title(cut_dep_name)
    plt.xlabel(cut_ind_name)
    plt.ylabel("")
    
    #%% Extract info from traces
    def get_min_ders(data):
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
    
    #%%
    #Fit Function
    Ffun = lambda w: np.log10((1+(1-3/(w**2))**(3/2)+9/(w**2))*w**3)
    Ffun2 = lambda w: np.log10((1-(1-3/(w**2))**(3/2)+9/(w**2))*w**3)
    start = 0
    end = 1600
    sf = freqs[800]
    fitfreqs = np.flip(freqs[start:end])+(1+np.sqrt(3))*sf
    plt.plot(np.flip(fitfreqs), Ffun((fitfreqs)), label = 'Rough "Fit" upper')
    plt.plot(np.flip(fitfreqs), Ffun2((fitfreqs)), label = 'Rough "Fit" lower')
    #%%
    plt.plot(get_min_ders(dep_var), ind_vars[0], label = 'forward')
    plt.plot(get_min_ders(dep_var2),ind_vars[0], label = 'backward')
    plt.legend(loc = 'right')
    plt.title("Minimum derivative points")
    plt.ylabel("VNA Power")
    plt.xlabel("Frequency")
    #%% process a pcolr (Duffing test plot)











