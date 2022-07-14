# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:53:23 2022

@author: Hatlab_3
"""
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import all_datadicts_from_hdf5

test_filepath = r'Z:/Data/NIST_Amp_Qubit_msmts/N25_L3_SP_2/raw_data_more_recs/2022-07-13/2022-07-13_0001_300.0ns_0.825amp/2022-07-13_0001_300.0ns_0.825amp.ddh5'

test_filepath = r'Z:/Data/NIST_Amp_Qubit_msmts/N25_L3_SP_2/raw_data_more_recs/2022-07-13/2022-07-13_0015_1000.0ns_0.425amp/2022-07-13_0015_1000.0ns_0.425amp.ddh5'


dd = all_datadicts_from_hdf5(test_filepath)['data']
recs = np.unique(dd['record_num']['values'])
times = np.unique(dd['time']['values'])
I_G = dd['I_G']['values'].reshape(recs.size, times.size)
Q_G = dd['Q_G']['values'].reshape(recs.size, times.size)
I_E = dd['I_E']['values'].reshape(recs.size, times.size)
Q_E = dd['Q_E']['values'].reshape(recs.size, times.size)
#%%
ones_arr = np.zeros(I_G.shape[1])
offset_start = 18
offset_end = 108
ones_arr[offset_start:offset_end] = 1
I_G_offset, Q_G_offset = np.average(I_G, axis = 0)[offset_start]*ones_arr, np.average(Q_G, axis = 0)[offset_start]*ones_arr
I_E_offset, Q_E_offset = np.average(I_E, axis = 0)[offset_start]*ones_arr, np.average(Q_E, axis = 0)[offset_start]*ones_arr

plt.figure()
plt.plot(times, np.average(I_E, axis = 0))
plt.plot(times, np.average(Q_E, axis = 0))
plt.plot(times, I_E_offset)
plt.plot(times, Q_E_offset)

plt.figure()
plt.plot(times, np.average(I_E, axis = 0)-I_E_offset)
plt.plot(times, np.average(Q_E, axis = 0)-Q_E_offset)

plt.figure()
plt.plot(np.average(I_E, axis = 0)-I_E_offset, np.average(Q_E, axis = 0)-Q_E_offset)
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

#%%
plt.figure()
plt.plot(np.average(I, axis = 0), np.average(Q, axis = 0))