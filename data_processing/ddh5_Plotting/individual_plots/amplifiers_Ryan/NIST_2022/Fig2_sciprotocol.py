# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:36:12 2022

@author: Ryan
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import proplot as pplt
import matplotlib.colors
from matplotlib import cm
import scipy.optimize as spo
import matplotlib as mpl
from matplotlib.colors import ListedColormap

cmap = mpl.cm.get_cmap('Div')
from_edge = 0.1
cnum = np.linspace(0+from_edge, 1-from_edge, 3)
newcolors = cmap(cnum)

greens = mpl.cm.get_cmap('greens')
greencmap = ListedColormap(greens(cnum))
# pink = np.array([248/256, 24/256, 148/256, 1])
# newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)
# newcmp = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
from cycler import cycler
default_prop_cycler = cycler('color', [newcmp(cnum[0]), newcmp(cnum[1]), newcmp(cnum[2])])
gain_colors = [newcmp(cnum[0]), greencmap(0.5), newcmp(cnum[2])]
plt.style.use('hatlab')
#supporting stuff courtesy of Boris
class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def hist2img(num_hist, den_hist, cutoff=1000):
    histogram = num_hist / den_hist

    histogram[np.where(den_hist < cutoff)] = np.nan

    imgR = histogram * 0
    imgG = histogram * 0
    imgB = histogram * 0

    imgR[np.where(np.isnan(histogram))] = 0.6
    imgG[np.where(np.isnan(histogram))] = 0.6
    imgB[np.where(np.isnan(histogram))] = 0.6

    COL = MplColorHelper('seismic', -2, 2)

    colormapR = COL.get_rgb(histogram)[:, :, 0]
    colormapG = COL.get_rgb(histogram)[:, :, 1]
    colormapB = COL.get_rgb(histogram)[:, :, 2]

    notNan = np.where(np.isnan(histogram) == False)

    weights = np.clip((den_hist / np.max(den_hist)) * 10000, 0, 1)

    imgR[notNan] = 0.5 * (1 - weights[notNan]) + colormapR[notNan] * weights[notNan]
    imgG[notNan] = 0.5 * (1 - weights[notNan]) + colormapG[notNan] * weights[notNan]
    imgB[notNan] = 0.5 * (1 - weights[notNan]) + colormapB[notNan] * weights[notNan]

    img = np.array([imgR, imgG, imgB])

    return img


def load_hist(dir, name, aligned=False):
    hf = ''
    if aligned:
        hf = h5py.File(dir + '\\' + name + '_hists_aligned.h5', 'r')
    else:
        hf = h5py.File(dir + '\\' + name + '_hists.h5', 'r')
    num_hists = np.array(hf.get('num_hists'))
    den_hists = np.array(hf.get('den_hists'))
    ampArray = np.array(hf.get('ampArray'))
    x = np.array(hf.get('x'))
    y = np.array(hf.get('y'))

    Imbar = 0
    Qmbar = 0
    sigma = 0

    if aligned:
        Imbar = np.array(hf.get('Imbar'))
        Qmbar = np.array(hf.get('Qmbar'))
        sigma = np.array(hf.get('sigma'))

    hf.close()

    if aligned:
        return num_hists, den_hists, ampArray, x, y, Imbar, Qmbar, sigma
    else:
        return num_hists, den_hists, ampArray, x, y

def linecut_fit(num_hists, den_hists, x, y, Imbar, Qmbar, sigma, args, cutoff=1, plot=True, constrained=False, ax = None):

    # note all of this assumes aligned data

    # constrained means: do we use Imbar/Qmbar from alignment step? Or do we leave them as free parameters to be fitted?

    tomo_axis = args[0]
    init_state = args[1]
    amp_i = args[2]
    direction = args[3]
    
    Imbar_sigma_i = np.sqrt((Imbar[amp_i]/sigma[amp_i]) ** 2 + (Qmbar[amp_i]/sigma[amp_i]) ** 2) # since assumed aligned (Imbar/Qmbar refer to unaligned values)

    def z_eta_f(Im_sigma, Imbar_sigma, alpha):
        return np.tanh(Im_sigma * Imbar_sigma + alpha)

    def z_eta_f_constrained(Im_sigma, alpha):
        return z_eta_f(Im_sigma, Imbar_sigma_i, alpha)

    def x_eta_f(Qm_sigma, Imbar_sigma, eta, theta, offset):
        # note this is only evaluated at Im=0
        # assumes histogram is already aligned: Qmbar = 0
        return np.sin(Qm_sigma * Imbar_sigma + theta) * np.exp(-Imbar_sigma ** 2 * (1 - eta) / eta) + offset

    def x_eta_f_constrained(Qm_sigma, eta, theta, offset):
        return x_eta_f(Qm_sigma, Imbar_sigma_i, eta, theta, offset)

    num_hist = num_hists[tomo_axis, init_state, amp_i, :, :]
    den_hist = den_hists[tomo_axis, init_state, amp_i, :, :]

    histogram = num_hist / den_hist

    histogram[np.where(den_hist < cutoff)] = np.nan

    m, n = np.shape(num_hist)

    result = 0

    m, n = np.shape(histogram)

    Im_sigma = x / sigma[amp_i]
    Qm_sigma = y / sigma[amp_i]

    res = []
    
    if ax is None: 
        fig, ax = pplt.subplots(nrows = 1, ncols = 1, hspace = '0.2em')

    if direction == 0:  # horizontal

        valid_idx = np.where(np.isnan(histogram[:, m // 2]) == False)

        sigmas = den_hist[valid_idx, m // 2][0]

        if constrained:

            res = spo.curve_fit(z_eta_f_constrained, Im_sigma[valid_idx], histogram[valid_idx, m // 2][0], p0=[0], sigma=sigmas)

            print('Constrained')
            print('Derived Imbar/sigma: ' + str(Imbar_sigma_i))
            print('Fit theta: ' + str(res[0][0]))
            print(' ')

            if plot:
                ax.plot(Im_sigma[valid_idx], histogram[valid_idx, m // 2][0], label='data')
                ax.plot(Im_sigma[valid_idx], z_eta_f(Im_sigma[valid_idx], Imbar_sigma_i, res[0][0]), label='partial fit')
                ax.legend()
        else:
            res = spo.curve_fit(z_eta_f, Im_sigma[valid_idx], histogram[valid_idx, m // 2][0], p0=[1, 0], sigma=sigmas)

            print(' ')

            print('Unconstrained')
            print('Fit Imbar/sigma: ' + str(res[0][0]))
            print('Fit theta: ' + str(res[0][1]))
            print(' ')

            if plot:
                ax.plot(Im_sigma[valid_idx], histogram[valid_idx, m // 2][0], '.', label='Data')

                ax.plot(Im_sigma[valid_idx], z_eta_f(Qm_sigma[valid_idx], res[0][0], res[0][1]), label='Fit')

    if direction == 1:  # vertical

        valid_idx = np.where(np.isnan(histogram[n // 2, :]) == False)

        sigmas = 1/np.sqrt(den_hist[n // 2, valid_idx][0]/np.max(den_hist))

        if constrained:

            res = spo.curve_fit(x_eta_f_constrained, Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], sigma=sigmas, p0=[0.5, 0, 0])

            print('Constrained')
            print('Derived Imbar/sigma: ' + str(Imbar_sigma_i))
            print('Fit Quantum Efficiency: ' + str(res[0][0]))
            print('Fit theta: ' + str(res[0][1]))
            print('Fit offset: ' + str(res[0][2]))
            print(' ')

            if plot:
                ax.plot(Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], label='data')

                ax.plot(Qm_sigma[valid_idx], x_eta_f(Qm_sigma[valid_idx], Imbar_sigma_i, res[0][0], res[0][1], res[0][2]), label='partial fit')
        else:



            print(sigmas)

            res = spo.curve_fit(x_eta_f, Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], sigma=sigmas, p0=[0.5, 0.5, 0, 0])

            print('Unconstrained')
            print('Fit Imbar/sigma: ' + str(res[0][0]))
            print('Fit Quantum Efficiency: ' + str(res[0][1]))
            print('Fit theta: ' + str(res[0][2]))
            print('Fit offset: ' + str(res[0][3]))
            print(' ')

            if plot:

                ax.plot(Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], '.', label='Data', color = gain_colors[0])

                ax.plot(Qm_sigma[valid_idx],
                         x_eta_f(Qm_sigma[valid_idx], res[0][0], res[0][1], res[0][2], res[0][3]), label='fit', color = gain_colors[1])
                x_ax = Qm_sigma[valid_idx]
                # ax.plot(Qm_sigma[valid_idx], den_hist[50][valid_idx])
                samples_ax = ax.panel('b')
                samples_ax.bar(x_ax, den_hist[valid_idx[0], 50], color = gain_colors[0])
                print('\n\n\ntotal_fit_samples: \n\n\n', np.sum(den_hist[valid_idx[0], 50]))

    return res, ax, samples_ax

def pepsi_plot(num_hists, den_hists, ampArray, x, y, initState=1, cutoff=100):
    
    gs = pplt.GridSpec(nrows = len(ampArray), ncols = 3, pad = 0, wspace = 0.2, hspace = 0.2)
    fig = pplt.figure(span=False, refwidth=1, share = False)

    axisArray = ['X', 'Y', 'Z']

    tomoAxisArray = [0, 1, 2]

    sum_den_hists = np.sum(den_hists, axis=0)

    for tomoAxis in tomoAxisArray:
        for amp_i in range(0, len(ampArray)):
            # ax0 = fig.subplot(gs[amp_i, 0])
            ax = fig.subplot(gs[amp_i, tomoAxis])
            #axs[amp_i, tomoAxis + 1]
            num_hist = num_hists[tomoAxis, initState, amp_i, :, :]
            den_hist = den_hists[tomoAxis, initState, amp_i, :, :]

            img = hist2img(num_hist, den_hist, cutoff=cutoff)

            im = ax.imshow(img.transpose(), extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            im_cb = im
            ax.set_xticks([])
            ax.set_yticks([])

            if tomoAxis == 0:
                den_trim = 30
                sum_den_hist = sum_den_hists[initState, amp_i, :, :]

                # ax0.pcolormesh(sum_den_hist.T[den_trim:-den_trim,den_trim:-den_trim], cmap='magma')
                print(np.shape(sum_den_hist))
                ax.set_ylabel(str(np.round(ampArray[amp_i]*1000, 0))+' mV')
            # ax0.set_xticks([])
            # ax0.set_yticks([])
            if amp_i == 0:
                ax.set_title(axisArray[tomoAxis])
                
    cbar = fig.colorbar(plt.get_cmap('seismic'), ticks = [0,1], location = 'r', title = r"$\langle X \rangle, \langle Y \rangle, \langle Z \rangle $ respectively")
    cbar.ax.set_yticklabels([-1, 1])
    return fig

#%%

fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/'
# fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/science_protocol_combined_hists.h5'
# fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_multiRep/science_protocol_multiRep_hists.h5'
name = r'science_protocol_combined'
res = load_hist(fp, name, aligned=True)
num_hists, den_hists, ampArray, x, y, Imbar, Qmbar, sigma = res

fig = pepsi_plot(*res[:5], cutoff = 0)
fig.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\science_protocol.pdf')

tomo_axis = 0
init_state = 1
amp_i = 3
direction = 1
#linecut_fit(num_hists, den_hists, x, y, Imbar, Qmbar, sigma, args, cutoff=1, plot=True, constrained=False)
fitfig, fitax = pplt.subplots()
res, lc_fit_ax, samples_ax = linecut_fit(num_hists, 
                             den_hists, 
                             x, y, 
                             Imbar, Qmbar, 
                             sigma, 
                             [tomo_axis, init_state, amp_i, direction], 
                             constrained = False, 
                             ax = fitax)
lc_fit_ax.legend(location = 'lc', ncols = 1, frame = 1)
ticklabelpad = mpl.rcParams['ytick.major.pad']

# samples_ax.annotate(r"$\overline{Q_m}$", xy=(1,0), xytext=(5, 0), ha='left', va='top',
#             xycoords='axes fraction', textcoords='offset points')
samples_ax.set_xlabel(r"$\overline{Q_m}$")
samples_ax.set_ylabel('Samples')
samples_ax.yaxis.tick_right()

lc_fit_ax.annotate(r"$\langle X \rangle$", xy=(-0.15,0.55), xytext=(-ticklabelpad, 5), ha='left', va='top',
            xycoords='axes fraction', textcoords='offset points')

# lc_fit_ax.set_ylabel(r"$\langle X \rangle$")
lc_fit_ax.set_yticks([-1,-0.5, 0, 0.5])
lc_fit_ax.set_yticklabels([-1,'', 0, 0.5])
lc_fit_ax.set

#%%
# print(np.shape(file['den_hists']))

# fig, axs = pplt.subplots(ncols = 3, nrows = 4, sharex = True, sharey = True)
# axs.set_aspect(1)
# axs.set_facecolor('gray')
# # axs.vstack()
# prep_to_plot = 1
# scale = 100
# for i in range(3):
#     for j in range(4): 
#         # Imbar, Qmbar = file['Imbar'][j], file['Qmbar'][j]
#         den_hist = file['den_hists'][i, prep_to_plot, j, :, :].T
#         num_hist = file['num_hists'][i, prep_to_plot, j, :, :].T

#         im = axs[i+3*j].pcolormesh(h_to_plot, cmap = 'seismic', vmin = -scale, vmax = scale, levels = 500)
# fig.colorbar(im, location = 'r', ticks = scale/4, tickminor = False)
#%%
