# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:31:47 2021

@author: Chao
"""

import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.optimize import curve_fit
import easygui
from plottr.data.datadict_storage import all_datadicts_from_hdf5

FREQ_UNIT = {'GHz' : 1e9,
             'MHz' : 1e6,
             'KHz' : 1e3,
             'Hz' : 1.0
             }


def rounder(value):
    return "{:.4e}".format(value)

def hangerFuncMagAndPhase(freq, Qext, Qint, f0, magBack, delta, phaseCorrect):
    omega0=f0
    
    x = (freq - omega0)/(omega0)
    S_21_up = Qext + 1j * Qext * Qint * (2 * x + 2 * delta / omega0)
    S_21_down = (Qint + Qext) + 2 * 1j * Qext * Qint * x

    S21 = magBack * (S_21_up / S_21_down) * np.exp(1j * (phaseCorrect)) #model by Kurtis Geerlings thesis
    
    mag = np.log10(np.abs(S21)) * 20
    fase = np.angle(S21)
    
    return (mag + 1j * fase).view(float)

freq = np.linspace(1.45e10, 1.55e10, 1600)
Qext = 2.5000e+04
Qint = 7.3073e+02
f0 = 1.5e10
magBack = 0.014113213258667646
delta =  18310403
phaseCorrect = -1.4392335945284946


test = hangerFuncMagAndPhase(freq, Qext, Qint, f0, magBack, delta, phaseCorrect)

plt.plot(freq, test[::2])

