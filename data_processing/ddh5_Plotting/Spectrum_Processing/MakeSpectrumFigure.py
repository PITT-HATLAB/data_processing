# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:39:24 2021

@author: Hatlab-RRK

Goal:
    
make a spectrum plot that can clearly show how a measurement is being done
"""

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('hatlab')

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

#%% plot with idler
fig, ax = plt.subplots()
x = np.linspace(-3, 3, 1000)
ax.fill_between(x, lorentzian(x, 0, 1, 3), alpha = 0.5, label = 'amplifier gain')
ax.fill_between(x, lorentzian(x,-1, 1, 6)+0.5, 0.5, alpha = 0.5, color = 'gray', label = 'detector band')
ax.fill_between(x, lorentzian(x,-1, 1, 0.5)+0.5, 0.5, alpha = 0.5, color = 'red', label = 'digital filter')
ax.set_ylim(0, 2)
ax.arrow(-1, 0, 0, 0.9, width = 0.05)
ax.arrow(1, 0, 0, 0.9, width = 0.05)
ax.arrow(0, 0.2, -1, 0, width = 0.03, length_includes_head = True, color = 'k')
# ax.grid()
ax.vlines(0, 0, 1, linestyle = 'dashed', color = 'red')
ax.set_xticks(ticks = [0])
ax.set_xticklabels(['$\omega_0 = 6GHz$'])
ax.get_yaxis().set_visible(False)
ax.annotate('Signal', [-2.2, 1.1])
ax.annotate('Idler', [1.1, 1.1])
ax.annotate('$\Delta = 1MHz$', [-0.5, 0.25], [-0.60, 0.3])
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# ax.annotate('Pump', [1.9, 0.96] )

#%%Qubit frequency stack
fig, ax = plt.subplots()
x = np.linspace(-3, 3, 1000)

ax.set_ylim(0, 2)
aw = 0.05
ax.arrow(-1, 0, 0, 0.9, width = aw, color = 'Blue')
# ax.arrow(-3, 0, 0, 0.9, width = aw)
ax.arrow(1, 0, 0, 0.9, width = aw)
ax.arrow(1, 0.2, -2, 0, width = 0.03, length_includes_head = True, color = 'k')
# ax.grid()
ax.vlines(0, 0, 1, linestyle = 'dashed', color = 'red')
ax.set_xticks(ticks = [-1,0,1])
ax.set_xticklabels(['$\omega_{a, 1}$', '$\omega_{drive}$', '$\omega_{a, 0}$'])
ax.get_yaxis().set_visible(False)
# ax.annotate('Signal', [-2.2, 1.1])
# ax.annotate('Idler', [1.1, 1.1])
ax.annotate('$\chi_{qc}$', [-0.75, 0.25], [-0.60, 0.3])
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# ax.annotate('Pump', [1.9, 0.96] )

