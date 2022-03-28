# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:08:20 2021

@author: Hatlab-RRK
"""
import pandas as pd
import numpy as np
import sys
R1 = 824.8
Rs = 822.3
Iin = 0.5e-3
inputVoltage = lambda dv: R1*Rs*Iin/np.array(dv)-R1-Rs

def fromVoltageCSV(filename): 
    volt_arr = pd.read_csv(filename, header = None)
    print(inputVoltage(volt_arr))

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])
    
