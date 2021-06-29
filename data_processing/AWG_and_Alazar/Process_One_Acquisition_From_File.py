# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:18:59 2021

@author: Ryan Kaufman - Hatlab
"""
from plottr.apps.autoplot import main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
from measurement_modules.AWG_and_Alazar.Pulse_Sweeping_utils import boxcar_histogram
import numpy as np
import matplotlib.pyplot as plt

datapath = r'E:/Data/Cooldown_20210611/SNAIL_Amps/C1/phase_preserving_checks/2021-06-16_0011_rotation_phase_3.77/2021-06-16_0011_rotation_phase_3.77.ddh5'

#autoplot, easiest way to see data if you dont need access to values 
# main(datapath, 'data')

#extracting individual arrays
dd = all_datadicts_from_hdf5(datapath)['data']

time_unit = dd['time']['unit']
time_vals = dd['time']['values'].reshape((7500, 208))

rec_unit = dd['record_num']['unit']
rec_num = dd['record_num']['values'].reshape((7500, 208))

I_plus = dd['I_plus']['values'].reshape((7500, 208))
I_minus = dd['I_minus']['values'].reshape((7500, 208))

Q_plus = dd['Q_plus']['values'].reshape((7500, 208))
Q_minus = dd['Q_minus']['values'].reshape((7500, 208))

#plotting averages
I_plus_avg = np.average(I_plus, axis = 0)
I_minus_avg = np.average(I_minus, axis = 0)
Q_plus_avg = np.average(Q_plus, axis = 0)
Q_minus_avg = np.average(Q_minus, axis = 0)

from measurement_modules.AWG_and_Alazar import Pulse_Sweeping_utils as PU
Gaussian_fits = []
#re-weave the data back into it's original pre-saved form

bins_even, bins_odd, h_even, h_odd = PU.Process_One_Acquisition(datapath.split('/')[-1], I_plus, I_minus, Q_plus, Q_minus, 55, 150, hist_scale = 0.02)
#%%
even_info_class = PU.fit_2D_Gaussian(bins_even, h_even)
odd_info_class = PU.fit_2D_Gaussian(bins_odd, h_odd)

xdata, ydata_even, ydata_odd = np.tile(bins_even[0:-1], 99), h_even.flatten(), h_odd.flatten()

from mpl_toolkits.axes_grid1 import make_axes_locatable
    
even_line_x, even_line_y = PU.get_contour_line(bins_even[:-1], bins_even[:-1], PU.Gaussian_2D(xdata,*even_info_class.info_dict['popt']).reshape(99,99))

odd_line_x, odd_line_y = PU.get_contour_line(bins_odd[:-1], bins_odd[:-1], PU.Gaussian_2D(xdata,*odd_info_class.info_dict['popt']).reshape(99,99))

fig, ax = plt.subplots()
pm = ax.pcolormesh(bins_even, bins_even, h_even+h_odd)
ax.plot(even_line_x, even_line_y, linestyle = '--', color = 'white')
ax.plot(odd_line_x, odd_line_y, linestyle = '--', color = 'white')
ax.set_xlabel('In-phase (V)')
ax.set_ylabel('Quadrature-phase (V)')
ax.set_title('100x100 bin Histogram')
ax.set_aspect(1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(pm, cax=cax, orientation='vertical')
ax.grid()

even_info_class.plot_on_ax(ax)
odd_info_class.plot_on_ax(ax)
even_info_class.print_info()
print('\n')
odd_info_class.print_info()
plt.show()
Gaussian_fits.append([even_info_class, odd_info_class])
    
# even_voltage = np.sqrt(np.sum(even_info_class.center_vec()**2))
# odd_voltage = np.sqrt(np.sum(odd_info_class.center_vec()**2))

#%%
S_off = Gaussian_fits[0]
S_on = Gaussian_fits[1]
mag_gain1 = np.linalg.norm(S_on[0].center_vec())/np.linalg.norm(S_off[0].center_vec())
mag_gain2 = np.linalg.norm(S_on[1].center_vec())/np.linalg.norm(S_off[1].center_vec())

print("Power gain 1 (dB): ", 20*np.log10(mag_gain1))
print("Power gain 2 (dB): ", 20*np.log10(mag_gain2))
avg_sigma_on = np.average(np.average(S_on[0].info_dict['sigma_x'], S_on[0].info_dict['sigma_y']))
print("avg_sigma_on/avg_sigma_off: ", S_on[0].info_dict['sigma_x'])

