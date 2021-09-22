# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:55:56 2021

@author: Hatlab-RRK
"""
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import numpy as np
import matplotlib.pyplot as plt

filepath = r'Z:/Data/SH_5B1_4141/pump_scanning/2021-09-21/2021-09-21_0008_First_test/2021-09-21_0008_First_test.ddh5'

Data = all_datadicts_from_hdf5(filepath)['data']
current = Data.extract('vna_return_power')['current']['values']
vna_input_power = Data.extract('vna_return_power')['vna_input_power']['values']
vna_return_powers = Data.extract('vna_return_power')['vna_return_power']['values']
gen_powers = Data.extract('vna_return_power')['gen_power']['values']
#take middle value for LO leakage
detuning = -1e6
#plot 
#%% make a 2D plot of vna_input_power on y-axis, gen_power on x_axis, vna_return power as color. Normalize to lowest gen power?
vna_input_size = np.size(np.unique(vna_input_power))
for bias_current in np.unique(current)[1:]: 
    current_filt = current == bias_current
    y_ax_gen_powers = np.unique(gen_powers)
    x_ax_vna_powers = np.unique(vna_input_power)
    vna_return_filt = vna_return_powers[current_filt].reshape(np.size(y_ax_gen_powers), np.size(x_ax_vna_powers))
    
    fig, ax = plt.subplots(figsize = (6,4))
    img = ax.pcolormesh(x_ax_vna_powers, y_ax_gen_powers, vna_return_filt)
    # ax.imshow(vna_return_filt)
    ax.set_xlabel('VNA input powers (dBm)')
    ax.set_ylabel('Generator Power (dBm)')
    ax.legend()
    ax.grid()
    ax.set_title(f'Current value: {bias_current*1000}mA')
    plt.colorbar(img, ax = ax)
    plt.show()
