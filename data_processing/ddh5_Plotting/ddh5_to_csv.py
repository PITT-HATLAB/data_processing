# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:46:46 2021

@author: Hatlab_3
"""
import h5py
import numpy as np
import csv

def ddh5_to_csv(ddh5_filepath, csv_filepath, csv_name): 
    ddh5_file = h5py.File(ddh5_filepath, 'r')
    data = ddh5_file['data']
    freqs = data['base_resonant_frequency']
    currents = data['current']
    qext = data['base_Qext']
    qexterror = np.sqrt(data['base_Qext_error'])
    qint = data['base_Qint']
    qinterror = np.sqrt(data['base_Qint_error'])
    with open(csv_filepath+'\\'+csv_name, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Current', 'Resonant Frequency', 'Qext', 'Qext Error', 'Qint', 'Qint Error'])
        writer.writeheader()
        for i, c in enumerate(currents): 
            writer.writerow({'Current': currents[i], 'Resonant Frequency': freqs[i], 'Qext': qext[i], 'Qext Error': qexterror[i], 'Qext Error': qexterror[i], 'Qint': qint[i], 'Qint Error': qinterror[i]})
    ddh5_file.close()
    
        
ddh5_to_csv(r'E:\Data\h5py_to_csv test\2021-02-22_0045_2021-02-22_0001_CPP_FS_fit.ddh5', r'E:\Data\h5py_to_csv test', 'CP2_fit.csv')