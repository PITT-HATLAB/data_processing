# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:39:24 2021

@author: Hatlab-RRK

Goal:
    
make a spectrum plot that can clearly show how a measurement is being done
"""

import matplotlib.pyplot as plt
import numpy as np

height_to_A = lambda h, fwhm: h*np.pi*fwhm
lorentzian = lambda x,x0,h,fwhm: (height_to_A(h, fwhm))/np.pi*(fwhm/((fwhm)**2+(x-x0)**2))
fig, ax = plt.subplots()
x = np.linspace(0, 2, 1000)
ax.fill_between(x, lorentzian(x, 0.5, 1, 0.1), alpha = 0.5)
ax.arrow(0.4, 0, 0, 0.9, width = 0.01)
ax.arrow(2, 0, 0, 0.9, width = 0.01)
ax.arrow(0.4, 0.5, 0.1, 0, width = 0.01, length_includes_head = True)
# ax.grid()
ax.vlines(.5, 0, 1, linestyle = 'dashed', color = 'red')
ax.set_xticks(ticks = [0.5, 2])
ax.set_xticklabels(['$\omega_0 = 6.8GHz$', '$2\omega_0$'])
ax.get_yaxis().set_visible(False)
ax.annotate('Signal', [0.2, 0.96])
ax.annotate('$\Delta = 100kHz$', [0.455, 0.5], [0.6, 0.7], arrowprops=dict(facecolor='black', width = 0.001, headwidth = 0.003) )
ax.annotate('Pump', [1.9, 0.96] )