# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:52:20 2022

@author: Hatlab-RRK
"""

#make a diagram of an anharmonic oscillator
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

x = np.linspace(-1,1, 100)
%matplotlib inline

harmonic = lambda m, x: m*x**2
plt.style.use('hatlab')
plt.plot(x, harmonic(0.5, x), label = 'Ground')
plt.plot(x, harmonic(1, x), color = 'k', label = 'Uncoupled')
plt.plot(x, harmonic(1.5, x), label = 'Excited')
plt.legend()

#%% Cavity frequency response
z0_over_zin = lambda omega, omega_res, kappa_in, kappa_out: (1j*omega/kappa_out)/(1+1j*omega/kappa_in-(omega/omega_res)**2)


gamma = lambda omega, omega_res, kappa_in, kappa_out: (1-z0_over_zin(omega, omega_res, kappa_in, kappa_out))/(1+z0_over_zin(omega, omega_res, kappa_in, kappa_out))
x = np.linspace(-100, 100, 100)
plt.figure()
plt.plot(x, np.abs(gamma(x, 500, 100, 10)**2))



plt.figure()


noise = 5
chi = 20

plt.plot(x, np.angle(gamma(x-chi/2, 500, 100, 10)))

# plt.fill_between(x, np.angle(gamma(x-chi/2+noise, 500, 100, 10)), np.angle(gamma(x-chi/2-noise, 500, 100, 10)), alpha = 0.5)

plt.plot(x, np.angle(gamma(x+chi/2, 500, 100, 10)))

# plt.fill_between(x, np.angle(gamma(x+chi/2-noise, 500, 100, 10)), np.angle(gamma(x+chi/2+noise, 500, 100, 10)), alpha = 0.5)

# plt.plot(x, np.angle(gamma(x+3*chi/2, 500, 100, 10)), label = '2nd excited')

# plt.fill_between(x, np.angle(gamma(x+3*chi/2-noise, 500, 100, 10)), np.angle(gamma(x+3*chi/2+noise, 500, 100, 10)), alpha = 0.5)

plt.title("Cavity Response")
plt.xlabel('Detuning (arb frequency units)')
plt.ylabel('Phase (rad)')
plt.vlines(0, np.pi, -np.pi, linestyle = 'dotted',colors = 'black')
plt.legend()
#%% Phasor diagram of g and e states
