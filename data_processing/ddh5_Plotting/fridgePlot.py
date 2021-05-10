# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:56:53 2021

@author: Hatlab_3
"""

import easygui
import os 
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
ddh5_filepath = r'\\169.254.29.187\Share\2021-01-12\2021-01-12_0002_Log_Test\2021-01-12_0002_Log_Test.ddh5'
fridge_dicts = all_datadicts_from_hdf5(ddh5_filepath)
fridgeDict = fridge_dicts['data']
mcDict = fridgeDict.extract('MC RuOx Temp (K)')

#get the arrays back out
mcTemp = mcDict.data_vals('MC RuOx Temp (K)')