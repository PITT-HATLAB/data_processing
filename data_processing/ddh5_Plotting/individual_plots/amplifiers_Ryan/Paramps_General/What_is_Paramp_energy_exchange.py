# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:23:14 2023

@author: Hatlab-RRK
"""

import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt 
U_C_complex = lambda C, V: 1/2*C*V**2
U_L_complex = lambda L, I: 1/2*L*I**2

L = lambda L0, dL, phi, t: L0+dL*np.cos(2*t+phi)
C = 2

tvals = np.linspace(0, 2*np.pi, 100)
It = np.cos(tvals)
Vt = np.sin(tvals)
dL = 0.5
L0 = 2
ind_L = L(L0, dL, np.pi, tvals)

U_C = U_C_complex(C, Vt)
U_L = U_L_complex(ind_L, It)

fig, axs = pplt.subplots(nrows = 1, ncols = 2, sharey = False, sharex = False)
fig.suptitle(f'$\delta L = {dL}$')
axs[0].plot(tvals/2/np.pi, U_C, label = '$U_C(t)$')
axs[0].plot(tvals/2/np.pi, U_L, label = '$U_L(t)$')
axs[0].plot(tvals/2/np.pi, U_L+U_C, 'k--', label = '$U_L(t)+U_C(t)$')
axs[0].set_ylabel("Energy (arb)")

axs[1].plot(tvals/2/np.pi, ind_L-L0, label = '$dL(t)$')
axs[1].plot(tvals/2/np.pi, np.abs(It), label = '$|I_L(t)$|')

axs.legend(loc = 'right', ncols = 1)
axs.set_xlabel("Time(ns)")

 