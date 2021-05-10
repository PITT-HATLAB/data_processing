# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:09:50 2021

@author: Hatlab_3

convert old-format hdf5 files into wolfie-format ddh5 files
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from plottr.data import datadict_storage as dds, datadict as dd
from plottr.apps.autoplot import autoplotDDH5, script, main

old_filepath = r'E:\Data\Cooldown_20210104\20210330_MariaChemPot_Processing\04_flux_sweep_-0.004_0.0mA_VNAfreq6500000000.0_9300000000.0GHz_Sss'
new_filepath = r'E:\Data\Cooldown_20210104\20210330_MariaChemPot_Processing'

#extract the data from the old file like we used to

old_file = h5py.File(old_filepath, 'r')
#%%
currents = old_file['currents'][()]
vna_freqs = old_file['measure_frequencies'][()]
vna_data = old_file['sweep_data'][()]
name = 'CP2_FS_fine_fit'

data = dd.DataDict(
        current = dict(unit='A'),
        frequency = dict(unit='Hz'),
        power = dict(axes=['current', 'frequency'], unit = 'dBm'), 
        phase = dict(axes=['current', 'frequency'], unit = 'rad'),
    )


with dds.DDH5Writer(new_filepath, data, name=name) as writer:
    for i, current in enumerate(currents): 
        freqs = vna_freqs
        vnadata = vna_data[i]
        writer.add_data(
                current = current*np.ones(np.size(freqs)),
                frequency = freqs,
                power = vnadata[0],
                phase = vnadata[1]/360*2*np.pi
            )
