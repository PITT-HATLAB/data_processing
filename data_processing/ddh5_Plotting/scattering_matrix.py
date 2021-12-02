# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:35:28 2021

@author: Hatlab-RRK
"""

from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
import os

def find_all_ddh5(cwd): 
    dirs = os.listdir(cwd)
    filepaths = []
    for path in dirs: 
        try:
            subs = os.listdir(cwd+'\\'+path)
            for sub in subs: 
                print(sub)
                if sub.split('.')[-1] == 'ddh5':  
                    filepaths.append(cwd+'\\'+path+'\\'+sub)
                else: 
                    for subsub in os.listdir(cwd+'\\'+path+'\\'+sub):
                        if subsub.split('.')[-1] == 'ddh5':  
                            filepaths.append(cwd+'\\'+path+'\\'+sub+'\\'+subsub)
        except: #usually because the files are one directory higher than you'd expect
            if path.split('.')[-1] == 'ddh5':  
                    filepaths.append(cwd+'\\'+path)
    return filepaths


smtx_dir = r'Z:\Data\SH6F1_1141\gp4\gc_noise_mtx1_2'
filepaths = find_all_ddh5(smtx_dir)
fig, axs = plt.subplots(2,2, figsize = (12,12))
axmap = {'S11': (0,0), 
         'S12': (0,1), 
         'S21': (1,0), 
         'S22': (1,1)}

cross_center = -85
cross_scale = 10
diag_center = -75
diag_scale = 15
fig.suptitle("GC1 (gp4)", fontsize = 40)

for filepath in filepaths:
    name = filepath.split('_')[-1].split('.')[0]
    ax = axs[axmap[name]]
    dd = all_datadicts_from_hdf5(filepath)['data']
    pows = dd.extract('power')['power']['values']
    freqs = dd.extract('power')['frequency']['values']
    
    #plot the LO leakage vs power

    ax.plot(freqs, pows)
    ax.set_xlabel('VNA frequency (GHz)')
    
    ax.set_ylabel('VNA Gain (dB)')
    ax.legend()
    ax.grid()
    if name[1] == name[2]: 
        ax.set_ylim(diag_center-diag_scale, diag_center+diag_scale)
    else: 
        ax.set_ylim(cross_center-cross_scale, cross_center+cross_scale)
    ax.set_title(name)