# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 13:27:49 2021

@author: Hatlab-RRK
"""
import numpy as np
import matplotlib.pyplot as plt
#visualizing phase offest in a mixer

I_0 = lambda t: np.cos(t)
Q_0 = lambda t: np.sin(t)

delay = np.pi/6
I_delayed = lambda t: np.cos(t+delay)

t = np.linspace(0, 2*np.pi-np.pi/32, 64, endpoint = False)

plt.plot(I_0(t), Q_0(t), label = 'base')


plt.plot(I_delayed(t), Q_0(t), label = 'I delayed')
plt.legend()
plt.gca().set_aspect(1)
#it skews the plot. Now compare the magnitudes
plt.figure()
plt.plot(t, I_0(t)**2+Q_0(t)**2, label = 'base')
plt.plot(t, I_delayed(t)**2+Q_0(t)**2, label = 'I delayed')
plt.legend()
plt.gca().set_aspect(1)
#notice the delayed version has higher maximum amplitude

#now we can try to correct the delay
I_corrected_by_delay = I_delayed(t-delay)

# def orthogonalization(I_delayed, Q_0): 
#     I_corrected, Q_corrected
#     for I_val, Q_val in I_delayed, Q_0: 
        

plt.figure()
plt.plot(I_0(t), Q_0(t), label = 'base')
plt.plot(I_delayed(t), Q_0(t), label = 'I delayed')
I_corrected = lambda t, Q, delay: (I_0(t)-Q*np.sin(delay))/np.cos(delay)
I_corrected_then_run_through_system = I_corrected(t-delay, Q_0(t), delay)
# plt.plot(I_corrected(t, Q_0(t), delay), Q_0(t), label = 'I corrected by function')
plt.plot(I_corrected_then_run_through_system, Q_0(t), label = 'I corrected by function then run through system')
# plt.plot(I_corrected_by_delay, Q_0(t), label = 'I corrected by delay')
plt.legend()
plt.gca().set_aspect(1)