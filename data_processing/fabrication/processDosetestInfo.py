# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:58:07 2021

@author: Hatlab-RRK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as color
from matplotlib.colors import ListedColormap

locationDict = {
    1111: (1,5), 
    1121: (2,5), 
    1131: (3,5),
    1141: (4,5),
    1151: (5,5),
    
    1112: (1,4), 
    1122: (2,4), 
    1132: (3,4),
    1142: (4,4),
    1152: (5,4),
    
    1211: (1,3), 
    1212: (2,3), 
    1221: (3,3),
    1231: (4,3), 
    1241: (5,3), 
    
    1222: (3,2),
    1232: (4,2), 
    1242: (5,2),
    
    1251: (4,1),
    1252: (5,1),
    
    
    2111: (1,6), 
    2121: (2,6), 
    2131: (3,6),
    2141: (4,6),
    2151: (5,6),
    
    2112: (1,7), 
    2122: (2,7), 
    2132: (3,7),
    2142: (4,7),
    2152: (5,7),
    
    2211: (1,8), 
    2212: (2,8), 
    2221: (3,8),
    2231: (4,8), 
    2241: (5,8), 
    
    2222: (3,9),
    2232: (4,9), 
    2242: (5,9),
    
    2251: (4,10),
    2252: (5,10),


    3111: (10,5), 
    3121: (9,5), 
    3131: (8,5),
    3141: (7,5),
    3151: (6,5),
    
    3112: (10,4), 
    3122: (9,4), 
    3132: (8,4),
    3142: (7,4),
    3152: (6,4),
    
    3211: (10,3), 
    3212: (9,3), 
    3221: (8,3),
    3231: (7,3), 
    3241: (6,3), 
    
    3222: (8,2),
    3232: (7,2), 
    3242: (6,2),
    
    3251: (7,1),
    3252: (6,1),
    
    4111: (10,6), 
    4121: (9,6), 
    4131: (8,6),
    4141: (7,6),
    4151: (6,6),
    
    4112: (10,7), 
    4122: (9,7), 
    4132: (8,7),
    4142: (7,7),
    4152: (6,7),
    
    4211: (10,8), 
    4212: (9,8), 
    4221: (8,8),
    4231: (7,8), 
    4241: (6,8), 
    
    4222: (8,9),
    4232: (7,9), 
    4242: (6,9),
    
    4251: (7, 10),
    4252: (6,10)
    
    }

optical_key = {
    'ggg': 'written language fails to describe how perfect this device is. (Literally nothing wrong)',
    'gg': 'really really good, like put that in the fridge ASAP (Very good, only minor imperfections)', 
    'g': 'good. could be a functional sample', 
    'o': 'ok. Like getting socks for christmas. Technically functional, but still disappointing (football-shaped)', 
    'b': 'bad. one bridge collapsed',
    'bb': 'really bad. Two bridges collapsed',
    'bbb': 'Lovecraft has nothing on this abomination'
    }
optical_val_key = {
    'ggg': 3,
    'gg': 2, 
    'g': 1, 
    'o': 0, 
    'b': -1,
    'bb': -2,
    'bbb': -3,
    'na': -3
    }

# dt_filepath = r'G:/My Drive/fab/20210812_SNAILs/SA_3X_SHARC_5X_msmts.csv'
dt_filepath = r'G:/My Drive/fab/20211119_SNAILs/SA_4X_SH_6X_msmts.csv'
data = pd.read_csv(dt_filepath)

ser_nums = data['Serial Number'][:-1]
l_or_s = data['extra on-chip index'][:-1]
res_vals = data['R_real'][:-1]
opt_vals  = data['optical_check'][:-1]


res_val_arr = np.empty(np.size(res_vals))
opt_val_arr = np.empty(np.size(res_vals))
loc_arr = []

for ind, SN in enumerate(ser_nums):
    print(SN, ind)
    loc_arr.append(locationDict[SN])
    if res_vals[ind] == '#DIV/0!':
        res_val_arr[ind] = -1
    else: 
        res_val_arr[ind] = res_vals[ind]
    # print(ind)
    # print(opt_vals[ind])
    opt_val_arr[ind] = optical_val_key[opt_vals[ind]]
    


# uncomment this if you have 2 SNAILS per chip instead of 1
# opt_imshow_arr = np.zeros((20,10))
# res_imshow_arr = np.zeros((20,10))
# for ind, loc in enumerate(loc_arr):
#     # print(ser_nums[ind])
#     if l_or_s[ind] == 's':
#         loc_corrected = (2*loc[0]-1-1, loc[1]-1)
#     if l_or_s[ind] == 'l': 
#         loc_corrected = (2*loc[0]-1, loc[1]-1)
#     # print(loc_corrected)
#     res_imshow_arr[loc_corrected] = res_val_arr[ind]
#     opt_imshow_arr[loc_corrected] = opt_val_arr[ind]
    
# uncomment this if you have 2 SNAILS per chip instead of 1
opt_imshow_arr = np.zeros((10,10))
res_imshow_arr = np.zeros((10,10))
for ind, loc in enumerate(loc_arr):
    # print(ser_nums[ind])
    if l_or_s[ind] == 's':
        loc_corrected = (2*loc[1]-1-1, loc[0]-1)
    if l_or_s[ind] == 'l': 
        loc_corrected = (2*loc[1]-1, loc[0]-1)
    else: 
        loc_corrected = (loc[1]-1, loc[0]-1)
    # print(loc_corrected)
    res_imshow_arr[loc_corrected] = res_val_arr[ind]
    opt_imshow_arr[loc_corrected] = opt_val_arr[ind]

colors = [color.hex2color('#4444FF'), color.hex2color('#FFFFFF'), color.hex2color('#05ff3f'), color.hex2color('#05ff3f'),color.hex2color('#FFFFFF'), color.hex2color('#FF4444')]
colors = [color.hex2color('#4444FF'), color.hex2color('#05ff3f'), color.hex2color('#05ff3f'), color.hex2color('#FF4444')]
_cmap = color.LinearSegmentedColormap.from_list('my_cmap', colors)
newcolors = _cmap(np.linspace(0,1,256))
gray = np.array([50/256, 50/256, 50/256, 1])
newcolors[:1, :] = gray
newcmp = ListedColormap(newcolors)

colors1 = [color.hex2color('#FF4444'),color.hex2color('#FFFFFF'), color.hex2color('#05ff3f')]
_cmap1 = color.LinearSegmentedColormap.from_list('my_cmap', colors1)
# print(_cmap1.N)
# _cmap1[0:10] == color.hex2color('FFFFFF')

x_ax = range(10)[1:]
y_ax = range(10)[1:]
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(121)
goal_resistance = 69

plt.pcolormesh(res_imshow_arr.T-goal_resistance, cmap = newcmp)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Resistance (Ohms)')
ax.invert_yaxis()
ax.set_aspect(1)
ax.hlines(np.arange(0,11,1), 0,10, linestyles = '-', colors = ['black'])
ax.vlines(np.arange(0,11,1), 0,10, linestyles = '-', colors = ['black'])
ax.set_title(f'Resistance measurements \ngoal_resistance: {goal_resistance} Ohms\n avg resistance (all): {np.round(np.average(res_val_arr[np.logical_not(res_val_arr==-1)]), 2)} Ohms\n avg resistance (opt gg or better): {np.round(np.average(res_val_arr[opt_val_arr >= 2]), 2)} Ohms')
plt.clim(-5,5)
for key, val in locationDict.items(): 
    ax.annotate(key, np.flip(np.array(val)+np.array([-0.5,-1])))
    
    

ax = fig.add_subplot(122)
plt.pcolormesh(opt_imshow_arr.T, cmap = _cmap1)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Optical Quality')
ax.invert_yaxis()
ax.set_aspect(1)
ax.hlines(np.arange(0,11,1), 0,10, linestyles = '-', colors = ['black'])
ax.vlines(np.arange(0,11,1), 0,10, linestyles = '-', colors = ['black'])
ax.set_title('Optical quality plotted wrt chip location')
plt.clim(-3,3)
for key, val in locationDict.items(): 
    ax.annotate(key, np.flip(np.array(val)+np.array([-0.5,-1])))