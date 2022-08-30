# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:53:23 2022

@author: Hatlab_3
"""
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from scipy.signal import savgol_filter as sf
#%%
test_filepath = r'Z:/Data/NIST_Amp_Qubit_msmts/N25_L3_SP_2_70us/raw_data/1us/2022-08-05/2022-08-05_0001_1000ns_0.075amp/2022-08-05_0001_1000ns_0.075amp.ddh5'
test_filepath = r'Z:/Data/NIST_Amp_Qubit_msmts/N25_L3_SP_2_70us/raw_data/1us/2022-08-05/2022-08-05_0002_1000ns_0.075amp/2022-08-05_0002_1000ns_0.075amp.ddh5'
test_filepath = r'G:/My Drive/shared/Amplifier_Response_Data/Data_20220805_80us_qubit/2022-08-05_0008_300ns_0.25amp.ddh5'
# test_filepath = r'Z:/Data/NIST_Amp_Qubit_msmts/N25_L3_SP_2/raw_data_more_recs/2022-07-13/2022-07-13_0015_1000.0ns_0.425amp/2022-07-13_0015_1000.0ns_0.425amp.ddh5'


dd = all_datadicts_from_hdf5(test_filepath)['data']
recs = np.unique(dd['record_num']['values'])
times = np.unique(dd['time']['values'])
I_G = dd['I_G']['values'].reshape(recs.size, times.size)
Q_G = dd['Q_G']['values'].reshape(recs.size, times.size)
I_E = dd['I_E']['values'].reshape(recs.size, times.size)
Q_E = dd['Q_E']['values'].reshape(recs.size, times.size)
#%%
ones_arr = np.zeros(I_G.shape[1])
sample_ind = 16
offset_start = 16
offset_end = 109

sf_args = {"window_length": 7, "polyorder": 1}
shim = 1.85
ones_arr[offset_start:offset_end] = 1
I_G_offset, Q_G_offset = sf(np.average(I_G, axis = 0)[sample_ind]*ones_arr, **sf_args)*shim, sf(np.average(Q_G, axis = 0)[sample_ind]*ones_arr, **sf_args)*shim
I_E_offset, Q_E_offset = sf(np.average(I_E, axis = 0)[sample_ind]*ones_arr, **sf_args)*shim, sf(np.average(Q_E, axis = 0)[sample_ind]*ones_arr, **sf_args)*shim

sf_args = {"window_length": 15, "polyorder": 3}

plt.figure()
plt.plot(times, np.average(I_E, axis = 0))
plt.plot(times, np.average(Q_E, axis = 0))
# plt.plot(times, I_E_offset)
# plt.plot(times, Q_E_offset)

# plt.figure()
# plt.plot(times, sf(np.average(I_E, axis = 0)-2*I_E_offset, **sf_args))
# plt.plot(times, sf(np.average(Q_E, axis = 0)-2*Q_E_offset, **sf_args))

# plt.figure()
# plt.plot(sf(np.average(I_E, axis = 0)-I_E_offset, **sf_args), sf(np.average(Q_E, axis = 0)-Q_E_offset, **sf_args))
# plt.plot(times, np.average(I_E, axis = 0)-I_E_offset)
# plt.plot(times, np.average(Q_E, axis = 0)-Q_E_offset)

# plt.figure()
# # plt.plot(np.average(I_G, axis = 0), 
# #           np.average(Q_G, axis = 0))
# plt.plot(np.average(I_E, axis = 0)-I_E_offset, 
#          np.average(Q_E, axis = 0)-Q_E_offset)

# plt.figure()
# for i, Is in enumerate(I):
#     if i%10 == 0: 
#         plt.plot(Is)
