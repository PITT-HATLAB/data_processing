# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:39:22 2020

@author: Tzu-Chiao Chien and Ryan Kaufman
"""


"""
Created on Mon Jul 24 19:45:51 2017

@author: Ryan Kaufman
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as color
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import easygui

cwd = easygui.diropenbox('Select where you are working')
print(cwd)
#%%
#Single File, single traces overlaid onto one-another if you choose: 
def get_plot(file = None):
    if file == None: 
        filepath = easygui.fileopenbox("Choose File", default = cwd)
        file = h5py.File(filepath)
        assert file != None
    datasets = file.keys()
    ind_var = easygui.choicebox("Pick the independent variable: ", choices = datasets)
    print(ind_var)
    dep_var = easygui.choicebox("Pick the dependent variable: ", choices = datasets)
    print(dep_var)
    
    ind_var_data = np.array(file[ind_var])
    dep_var_data = np.array(file[dep_var])
    
    
    return ind_var_data, dep_var_data, ind_var, dep_var

ind_vars, dep_vars, ind_name, dep_name = get_plot()
# ind_vars2, dep_vars2, ind_name2, dep_name2 = get_plot()
#%%
plt.plot(ind_vars, dep_vars)
#%%
def change_unit(name_string, new_unit): 
    return(name_string[0:(name_string.index('(')+1)]+new_unit+')')
#%%

offset_right = -32
offset_left = 10
fitx = np.power(10,(ind_vars)/20)[offset_left:offset_right]
fitx = (fitx-fitx[0])*np.sqrt(1000)
fity = np.power(10,(dep_vars-dep_vars[1])/20)[offset_left:offset_right]
# print("Max: {:.3f} \nMin: {:.3f}".format(np.max(dep_vars), np.min(dep_vars)))
plt.close("all")
plt.plot(fitx, fity,'o')
# plt.plot(ind_vars2[0], dep_vars2[0], label = "Backward")
plt.legend()
ind_name = change_unit(ind_name,'mV')
dep_name = change_unit(dep_name,'$\\frac{V}{V}$')
plt.xlabel(ind_name)
plt.ylabel(dep_name)
plt.title("Gain (V/V) vs.Pump power (V)")

#%% Numpy Fit
from numpy.polynomial.polynomial import polyfit
fit = np.flip(np.polyfit(fitx,fity,2))
fit_func = lambda x,a,b,c: a+b*x+c*x**2
english_sucks = ["th","st","nd","rd","th"]
for coefficient in enumerate(fit):
    print("{}{} order term: {}".format(coefficient[0],english_sucks[coefficient[0]],coefficient[1]))
plt.plot(fitx,fity,'o')
plt.plot(fitx,fit_func(fitx, *fit))
plt.xlabel(ind_name)
plt.ylabel(dep_name)
plt.title("All Order Fit")
#%% Scipy Fit
from scipy.optimize import curve_fit
def fit_func(x,b): 
    return b*x**2
fit_end, other_stuff = curve_fit(fit_func, fitx,fity)
print("Second order: {} (MHz/mV)^2".format(fit_end[0]))
plt.plot(fitx,fity,'o')
plt.plot(fitx,fit_func(fitx,fit_end[0]))
plt.xlabel("Linearized Voltage ($\sqrt{\mu W}$)")
plt.ylabel("Gain ()")
plt.title("2nd order fit")
#%% Fit to lorentzian
from scipy.optimize import curve_fit
def double_lorentzian(x,a,b,c): 
    return ((a**2+(x-c)**2)/(b**2-(x-c)**2))**2

fit_end, other_stuff = curve_fit(double_lorentzian, fitx,fity, [6,5,0])
plt.plot(fitx,fity,'o')
plt_vals = np.linspace(np.min(fitx), np.max(fitx), 150)
plt.plot(plt_vals,double_lorentzian(plt_vals,fit_end[0], fit_end[1], fit_end[2]))
print(fit_end)
plt.ylim(0,25)
plt.xlabel("Linearized Voltage ($\sqrt{\mu W}$)")
plt.ylabel("VNA Gain (V/V)")
plt.title("Lorentzian fit")

#%%Plot all values
offset_right = -1
offset_left = 0
fitx = np.power(10,(ind_vars)/20)[offset_left:offset_right]
fitx = (fitx-fitx[0])*np.sqrt(1000)
fity = np.power(10,(dep_vars-dep_vars[1])/20)[offset_left:offset_right]
plt_vals = np.linspace(np.min(fitx), np.max(fitx), 100)
plt.plot(fitx,fity,'o')
plt.plot(plt_vals,double_lorentzian(plt_vals,fit_end[0], fit_end[1], fit_end[2]))
plt.ylim(0,25)







