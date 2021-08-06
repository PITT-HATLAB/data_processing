# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:35:29 2021

@author: Hatlab-RRK
"""

import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
from data_processing.models.SNAIL_supporting_modules.Participation_and_Alpha_Fitter import slider_fit
from data_processing.fitting.QFit import fit, plotRes
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from data_processing.Helper_Functions import find_all_ddh5
from data_processing.ddh5_Plotting.TACO_multiplot_b1 import superTACO_Bars
import pandas as pd
from scipy.optimize import curve_fit



class hamiltonian(): 
    
    def __init___(self, symbols, energy_equation): 
        
    
    
    
def get_phi_min_funcs(alpha, phi_ext_arr): 
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail_norm = -a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3)
    c1 = sp.series(U_snail_norm, phi_s, x0 = phi_m, n = 2).removeO().coeff((phi_s-phi_m))
    #generate a lambda function that outputs another lambda function for a given phi_ext
    #which then depends on phi_m only
    func_arr = []
    for phi_ext in phi_ext_arr: 
        c1_num = sp.lambdify(phi_m, c1.subs(a, alpha).subs(phi_e, phi_ext), "numpy")
        func_arr.append(c1_num)
    return func_arr
def get_phi_min_fsolve(alpha, phi_ext_arr):
    funcs = get_phi_min_funcs(alpha, phi_ext_arr)
    sol_arr = np.ones(np.size(funcs))
    for i, func in enumerate(funcs): 
        sol_arr[i] = fsolve(func, phi_ext_arr[i])
    return sol_arr
def get_phi_min(alpha, phi_ext): 
    func = get_phi_min_funcs(alpha, [phi_ext])[0]
    return(fsolve(func, phi_ext)[0])

def c4_func_gen_vectorize(alpha_val): #can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 5)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 4))
    c4exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c4exp)

def c3_func_gen_vectorize(alpha_val): #can be fed an array
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 4)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 3))
    c3exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c3exp)

def c2_func_gen_vectorize(alpha_val): 
    a, Ej, phi_s, phi_e, phi_m = sp.symbols('alpha,E_j,phi_s,phi_e, phi_min')
    U_snail = (-a*sp.cos(phi_s) - 3*sp.cos((phi_e-phi_s)/3))
    expansion = sp.series(U_snail, phi_s, x0 = phi_m, n = 3)
    coeff = expansion.removeO().coeff(sp.Pow(phi_s-phi_m, 2))
    c2exp = lambda phi_ext: coeff.subs([(a, alpha_val), (phi_e, phi_ext), (phi_m, get_phi_min(alpha_val, phi_ext))])
    return np.vectorize(c2exp)