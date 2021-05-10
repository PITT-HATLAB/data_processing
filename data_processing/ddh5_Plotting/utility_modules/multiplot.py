# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:02:03 2021

@author: Hatlab_3

for getting many files out of a particular directory with Wolfie's directory structure'
"""
import easygui
import os 
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5


cwd = r'E:\Data\Cooldown_20210104\-0.173mA_transition\2021-02-09'


def find_all_ddh5(cwd): 
    dirs = os.listdir(cwd)
    filepaths = []
    for path in dirs: 
        rechecks = []
        subs = os.listdir(cwd+'\\'+path)
        for sub in subs: 
            print(sub)
            if sub.split('.')[-1] == 'ddh5':  
                filepaths.append(cwd+'\\'+path+'\\'+sub)
            else: 
                for subsub in os.listdir(cwd+'\\'+path+'\\'+sub): 
                    if subsub.split('.')[-1] == 'ddh5':  
                        filepaths.append(cwd+'\\'+path+'\\'+sub+'\\'+subsub)
    return filepaths

res = find_all_ddh5(cwd)
for filename in res: 
    main(filename, 'data')