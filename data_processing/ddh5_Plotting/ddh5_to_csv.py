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
    
        
ddh5_to_csv(r'X:/Data/HuntCollab2/hBN_cap/20210823_cooldown/2021-08-30/2021-08-30_0002_13_838_2/2021-08-30_0002_13_838_2.ddh5', 'X:/Data/HuntCollab2/hBN_cap/20210823_cooldown/2021-08-30/2021-08-30_0002_13_838_2', '13_838_ghz_reso.csv')