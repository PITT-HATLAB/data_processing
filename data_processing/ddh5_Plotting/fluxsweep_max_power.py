# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target
from data_processing.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.fftpack import dct, idct


#%%





def leakage(foward_filename, reverse_filename):
    
    dd = all_datadicts_from_hdf5(filepath)['data']
    powers_dB = dd.extract('power')['power']['values']
    freqs = dd.extract('power')['frequency']['values']*2*np.pi
    phase_rad = dd.extract('power')['phase']['values']
    
    
    plt.semilogx(np.reshape(dd['voltage']['values'],[50,1001])[:,596], np.reshape(dd['power']['values'],[50,1001])[:,596])
    
    
# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target
from data_processing.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.fftpack import dct, idct


def fluxsweep_max_power(filename, shape, mode_boundary):
    '''
    Looks at a flux sweep and pulls out the frequency and power at the maximum
    power point for each current bias. For 2-mode sweeps.

    Parameters
    ----------
    filename : string
        path to flux sweeep.
        
    shape : tuple
        shape to unpack the data correctly
        
    mode_boundary : float
        the frequency above which is one mode, below is the other

    Returns
    -------
    None.

    '''


    dd = all_datadicts_from_hdf5(filepath)['data']
    powers_dB = dd.extract('power')['power']['values']
    freq = dd.extract('power')['frequency']['values']*2*np.pi
    current = dd.extract('power')['current']['values']
    
    powers_dB = np.reshape(powers_dB, shape)
    freq = np.reshape(freq, shape)
    current = np.reshape(current, shape)
    
    freq = freq[0,:]
    current = current[:,0]
    
    max_power_S = current*0
    max_power_A = current*0
    max_freq_S = current*0
    max_freq_A = current*0
    
    
    S_i = np.where(freq < mode_boundary)
    A_i = np.where(freq > mode_boundary)
    
    for i in range(0,len(current)):
        
        powers_dB_row_S = np.squeeze(powers_dB[i,S_i])
        powers_dB_row_A = np.squeeze(powers_dB[i,A_i])
        
        
        max_freq_S[i] = np.mean(freq[np.where(powers_dB_row_S == np.max(powers_dB_row_S))])
        max_freq_A[i] = np.mean(freq[np.where(powers_dB_row_A == np.max(powers_dB_row_A))])
        
        max_power_S[i] = np.mean(powers_dB_row_S[np.where(powers_dB_row_S == np.max(powers_dB_row_S))])
        max_power_A[i] = np.mean(powers_dB_row_A[np.where(powers_dB_row_A == np.max(powers_dB_row_A))])


    return current, max_power_S, max_power_A, max_freq_S, max_freq_A

filepath = r'//136.142.53.51/data001/Data/SH_5B1/fluxsweep/SNAIL/2021-09-27/2021-09-27_0002_transmission_flux_sweep_-35dBm/2021-09-27_0002_transmission_flux_sweep_-35dBm.ddh5'
mode_boundary = 7e9 * 2 * np.pi    
shape = (600,500)

current, max_power_S, max_power_A, max_freq_S, max_freq_A = fluxsweep_max_power(filepath, shape, mode_boundary)
    
plt.figure(1)
plt.clf()
plt.plot(current,max_freq_S/(2 * np.pi))
plt.plot(current,max_freq_A/(2 * np.pi))

plt.figure(2)
plt.clf()
plt.plot(current,max_power_S/(2 * np.pi))
plt.plot(current,max_power_A/(2 * np.pi))