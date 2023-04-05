# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:19:22 2021

@author: Hatlab-RRK
"""
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#youngs modulus of Brass
Ybrass = 97e9 #pa

#mechanical efficiency of brass screw (empirical)
eta_Brass_256 = 0.0047

#lead of the brass screw (distance between adjacent threads)
lead_Brass_256 = 0.452e-3#m

#cross sectional area of the screw
A_Brass_256 = 9.08e-6#m^2, from minor diameter

#x-axis
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
torque_in_arr = np.linspace(0, 120) #in-oz
torque_in_arr_metric = torque_in_arr*inoz_to_Nm

sig_out_arr = torque_in_arr*inoz_to_Nm*2*np.pi/lead_Brass_440/A_Brass_440/1e6
fig, ax = plt.subplots()
ax.plot(torque_in_arr, sig_out_arr)

ax.set_xlabel("Torque(in-oz)")
ax.set_ylabel("Stress (MPa)")
ax.grid()
ax.hlines([140], 0, 120, label = "Yield Strength (Brass, MPa)", linestyle = 'dashed', color = 'orange')
ax.hlines([360], 0, 120, label = "Tensile Strength (Brass, MPa)", linestyle = 'dashed', color = 'red')
ax.legend()
ax.set_title("4-40 Brass")

#%%Differential thermal strain image plot
alpha_Br = -384e-5
alpha_Mo = -95e-5
alpha_Al = -414.9e-5
alpha_SS = -298.6e-5
alpha_Cu = -326e-5
alpha_tef = -2127e-5

YBrass = 125e9
YSS = 193e9
YAl = 68e9
YCu = 110e9
YMo = 329e9
Ytef = 575e6


# L1_arr = np.linspace(0, 5, 155)*10#mm, 0-5cm 
# L2_arr = np.linspace(5.1, 5.3, 100) #mm #length of washer
L2_arr = np.linspace(-150, 5.3, 100) #mm #length of washer
sRT_arr = np.linspace(0, 150, 100)*1e6

md = {'brass': {'Y': YBrass, 'alpha': alpha_Br}, 
      'SS': {'Y': YSS, 'alpha': alpha_SS}, 
      'Al': {'Y': YAl, 'alpha': alpha_Al}, 
      'Mo': {'Y': YMo, 'alpha': alpha_Mo}, 
      'Cu': {'Y': YCu, 'alpha': alpha_Cu}, 
      'Teflon': {'Y': Ytef, 'alpha': alpha_tef}
      }

sigma_cryo = lambda L0, L1, L2, a0, a1, a2, Y0, sRT: -(a2*L2+a1*L1-a0*L0)/(a0*L0)*Y0+sRT #equation from powerpoint slide

screw_mat = 'SS'
bulk_mat = 'SS'
washer_mat = 'SS'

sCryo = []
# L1 = 10 #mm
L1 = 155
for L2 in L2_arr: 
    sCryo_piece = []
    for sRT in sRT_arr: 
        L0 = L1+L2
        sCryo_piece.append(sigma_cryo(L0, L1, L2, md[screw_mat]['alpha'], md[bulk_mat]['alpha'], md[washer_mat]['alpha'], md[screw_mat]['Y'], sRT))
        # sCryo_piece.append(sigma_cryo(L0, L1, L2, alpha_SS, alpha_Cu, alpha_Mo, YSS, sRT))
    sCryo.append(sCryo_piece)

im = plt.pcolormesh(L2_arr, sRT_arr/1e6, np.array(sCryo).T/1e6, cmap = 'seismic', vmin = -124, vmax = 124)

plt.title("")
plt.xlabel(f"length_of_{washer_mat} (mm)")
plt.ylabel("$\sigma_{RT}$ (MPa)")
plt.title(f"Stress (MPa) vs tightening and {washer_mat} Length, L1 = {L1} mm")
cb = plt.colorbar()

# plt.figure()
# sCryo = []
# L1 = 3 #mm
# # L1 = 25
# for L2 in L2_arr: 
#     sCryo_piece = []
#     for sRT in sRT_arr: 
#         L0 = L1+L2
#         sCryo_piece.append(sigma_cryo(L0, L1, L2, alpha_SS, alpha_Cu, alpha_Mo, YSS, sRT))
        
#     sCryo.append(sCryo_piece)

# im = plt.pcolormesh(sRT_arr/1e6, L2_arr, np.array(sCryo)/1e6, cmap = 'seismic', vmin = -360, vmax = 360)

# plt.title("")
# plt.ylabel("length_of_Mo (mm)")
# plt.xlabel("$\sigma_{RT}$ (MPa)")
# plt.title("Stress (MPa) vs tightening and Mo Length")
# cb = plt.colorbar()













