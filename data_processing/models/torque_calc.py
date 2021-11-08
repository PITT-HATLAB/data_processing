# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:19:22 2021

@author: Hatlab-RRK
"""
import numpy as np
import matplotlib.pyplot as plt

Ybrass = 97e9 #pa
eta_Brass_256 = 0.0047
lead_Brass_256 = 0.452e-3#m
A_Brass_256 = 9.08e-6#m^2, from minor diameter

torque_in_arr = np.linspace(0, 60) #in-oz
inoz_to_Nm = 1/141.611

torque_in_arr_metric = torque_in_arr*inoz_to_Nm
fig, ax = plt.subplots()
# F_out_arr = eta_Brass_256*(2*np.pi/lead_Brass_256)*torque_in_arr_metric
# sig_out_arr= F_out_arr/A_Brass_256
# 
# ax.plot(torque_in_arr, F_out_arr)

sig_out_arr = torque_in_arr*inoz_to_Nm*2*np.pi/lead_Brass_256/A_Brass_256/1e6
ax.plot(torque_in_arr, sig_out_arr)
ax.set_xlabel("Torque(in-oz)")
ax.set_ylabel("Stress (MPa)")
ax.grid()
ax.hlines([140], 0, 60, label = "Yield Strength (Brass, MPa)", linestyle = 'dashed', color = 'orange')
ax.hlines([360], 0, 60, label = "Tensile Strength (Brass, MPa)", linestyle = 'dashed', color = 'red')
ax.legend()
ax.set_title("2-56 Brass")

#%% assuming same mechanical efficiency for 4-40
lead_Brass_440 = 0.635/1e3 #m
A_Brass_440 = (np.pi*2.38506**2)/1e6 #m^2

torque_in_arr_metric = torque_in_arr*inoz_to_Nm

F_out_arr = eta_Brass_256*(2*np.pi/lead_Brass_440)*torque_in_arr
sig_out_arr= F_out_arr/A_Brass_440
fig, ax = plt.subplots()
ax.plot(torque_in_arr, sig_out_arr)

ax.set_xlabel("Torque(in-oz)")
ax.set_ylabel("Stress (MPa)")
ax.grid()
ax.hlines([310e6], 0, 60, label = "Yield Strength (Brass)", linestyle = 'dashed', color = 'red')
ax.legend()
ax.set_title("2-56 Brass")