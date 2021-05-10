# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:06:10 2021

@author: Hatlab_3
"""
import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from hat_utilities.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target

from scipy.ndimage import gaussian_filter
fridge_filepath = r'\\169.254.29.187\Share\2021-01-12\2021-01-12_0002_Log_Test\2021-01-12_0002_Log_Test.ddh5'
flux_filepath = r'H:\Data\Fridge Texas\Cooldown_20210104\SHARC41\2021-01-12\2021-01-12_0007_2G_FS_Snail+B_mode_2-narrower_-10dBm_VNA\2021-01-12_0007_2G_FS_Snail+B_mode_2-narrower_-10dBm_VNA.ddh5'


#%% Autoplotting
main(flux_filepath, 'data')
#%%
main(fridge_filepath, 'data')