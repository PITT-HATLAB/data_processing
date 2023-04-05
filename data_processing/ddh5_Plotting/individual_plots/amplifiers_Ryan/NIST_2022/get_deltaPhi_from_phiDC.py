# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:48:39 2023

@author: Hatlab-RRK
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

rootfig, rootax = plt.subplots()

dphi_arr = np.linspace(0,np.pi)

LHS = lambda dphi, phiExt : 60/15*(dphi-phiExt)
RHS = lambda dphi : np.sin(dphi)

LHS_pi2 = lambda dphi: LHS(dphi, np.pi/2)

rootax.plot(dphi_arr/np.pi, LHS_pi2(dphi_arr), label = 'LHS')
rootax.plot(dphi_arr/np.pi, RHS(dphi_arr), label = 'RHS')

rootfunc_pi2 = lambda dphi: LHS_pi2(dphi)-RHS(dphi)

res = root(rootfunc_pi2, np.pi/2)
xval = np.round(res['x'][0], 3)
rootax.plot(xval/np.pi, RHS(xval), 'r.', label = f'Root at {np.round(xval/np.pi, 3)}pi')
rootax.legend()
rootax.set_ylim(0,1.5)

# phiDC_fig, phiDC_ax = plt.subplots()

#now repeat that for a bunch of phiDC's at a given shunting ratio