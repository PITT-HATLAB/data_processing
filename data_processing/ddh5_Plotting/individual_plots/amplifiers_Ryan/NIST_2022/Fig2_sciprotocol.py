# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:36:12 2022

@author: Ryan
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import scipy.optimize as spo
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import warnings
import proplot as pplt

cmap = cm.get_cmap('Div')
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
# plt.style.use('hatlab')
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
    # these are the boundaries, I need the centers
    x = np.array(hf.get('x'))
    x_ret = x[:-1]+(x[1]-x[0])/2
    y = np.array(hf.get('y'))
    y_ret = x[:-1] + (x[1] - x[0]) / 2

    Imbar = 0
    Qmbar = 0
    sigma = 0

    if aligned:
        Imbar = np.array(hf.get('Imbar'))
        Qmbar = np.array(hf.get('Qmbar'))
        sigma = np.array(hf.get('sigma'))

    hf.close()

    if aligned:
        return num_hists, den_hists, ampArray, x_ret, y_ret, Imbar, Qmbar, sigma
    else:
        return num_hists, den_hists, ampArray, x_ret, y_ret

def linecut_fit(num_hists,
                den_hists,
                x, y,
                Imbar, Qmbar,
                sigma,
                args, cutoff=1,
                plot=True, constrained=False,
                ax = None, plot_samples = False,
                trim_left = 0, trim_right = -1):

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

    def xy_eta_f(Qm_sigma, Imbar_sigma, eta, thetaX, thetaY, offset):
        #only evaluated at Imbar = 0
        #assumes the first half of the data is X back-action, last half is y (with only a different theta allowed)
        return np.append(np.sin(Qm_sigma * Imbar_sigma + thetaX) * np.exp(-Imbar_sigma ** 2 * (1 - eta) / eta) + offset,
                         np.sin(Qm_sigma * Imbar_sigma + thetaY) * np.exp(-Imbar_sigma ** 2 * (1 - eta) / eta) + offset
                         )

    def x_eta_f_constrained(Qm_sigma, eta, theta, offset):
        return x_eta_f(Qm_sigma, Imbar_sigma_i, eta, theta, offset)

    num_hist = num_hists[tomo_axis, init_state, amp_i, :, :]
    den_hist = den_hists[tomo_axis, init_state, amp_i, :, :]

    print('num hist sum', np.sum(num_hist))
    print('den hist sum', np.sum(den_hist))
    print('num hist max', np.max(num_hist))
    print('den hist max', np.max(den_hist))

    histogram = num_hist / den_hist

    histogram[np.where(den_hist < cutoff)] = np.nan

    m, n = np.shape(num_hist)

    result = 0

    m, n = np.shape(histogram)

    #plot the other axis too
    num_hist2 = num_hists[tomo_axis+1, init_state, amp_i, :, :]
    den_hist2 = den_hists[tomo_axis+1, init_state, amp_i, :, :]

    histogram2 = num_hist2 / den_hist2

    histogram2[np.where(den_hist2 < cutoff)] = np.nan

    m2, n2 = np.shape(num_hist2)

    result = 0

    m2, n2 = np.shape(histogram2)

    Im_sigma = x / sigma[amp_i]
    Qm_sigma = y / sigma[amp_i]

    print("shape of x, y: ", np.shape(x), np.shape(y))

    #have to make sure the shape of Qm_sigma matches the shape of the histogram n//2



    res = []

    if ax is None:
        fig, ax = pplt.subplots(nrows = 1, ncols = 1, hspace = '0.2em')

    if direction == 0:  # horizontal
        pass
    if direction == 1:  # vertical
        # valid_idx = np.where(np.isnan(histogram[n // 2, :]) == False)
        # valid_idx2 = np.where(np.isnan(histogram2[n2 // 2, :]) == False)

        # valid_idx = np.ones_like(histogram[n // 2, :]).astype(bool)
        # valid_idx2 = np.ones_like(histogram2[n2 // 2, :]).astype(bool)
        valid_idx = np.isnan(histogram[n // 2, :]) == False
        valid_idx2 = np.isnan(histogram2[n2 // 2, :]) == False

        print("valid idx shape", valid_idx.shape, "sum: ", np.sum(valid_idx))
        print("valid idx2 shape", valid_idx2.shape, "sum: ", np.sum(valid_idx2))

        '''
        the fit function requires the validated slices to be the same length, so we have to trim the longer one
        by constraining it to where the other is also valid
        '''
        valid_idx_total = np.logical_and(valid_idx, valid_idx2)
        print("valid idx total shape", valid_idx_total.shape, 'sum: ', np.sum(valid_idx_total))

        x_slice = histogram[(n) // 2, valid_idx_total]
        y_slice = histogram2[(n2) // 2, valid_idx_total]
        num_shots = den_hist[n // 2, valid_idx_total]
        num_shots2 = den_hist2[n // 2, valid_idx_total]
        # print('number of shots in each bin', num_shots)
        plt.figure()
        plt.plot(num_shots)
        plt.plot(num_shots2)
        sigmas = 1 / np.sqrt(den_hist[n // 2, valid_idx_total] / np.max(den_hist[n // 2, valid_idx_total]))
        sigmas_plot = np.abs(sigmas) / np.sqrt(num_shots)
        print("sigmas shape: ", sigmas.shape)
        sigmas2 = 1 / np.sqrt(den_hist2[n2 // 2, valid_idx_total] / np.max(den_hist2[n2 // 2, valid_idx_total]))
        sigmas2_plot = np.abs(sigmas2) / np.sqrt(num_shots2)
        print("sigmas2 shape: ", sigmas2.shape)

        if constrained:
            pass
            # res = spo.curve_fit(x_eta_f_constrained, Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], sigma=np.append(sigmas, sigmas2), p0=[0.5, 0, 0])

            # print('Constrained')
            # print('Derived Imbar/sigma: ' + str(Imbar_sigma_i))
            # print('Fit Quantum Efficiency: ' + str(res[0][0]))
            # print('Fit theta: ' + str(res[0][1]))
            # print('Fit offset: ' + str(res[0][2]))
            # print(' ')

            # if plot:
            #     ax.plot(Qm_sigma[valid_idx], histogram[n // 2, valid_idx][0], label='data')

            #     ax.plot(Qm_sigma[valid_idx], x_eta_f(Qm_sigma[valid_idx], Imbar_sigma_i, res[0][0], res[0][1], res[0][2]), label='partial fit')
        else:



            # print(sigmas)

            print("shape of sigmas: ", np.shape(sigmas), np.shape(sigmas2))
            print("shape of sigmas after trimming: ",
                  np.shape(sigmas[trim_left:trim_right]),
                  np.shape(sigmas2[trim_left:trim_right]))

            print("shape of the Qm's pre validation: ", np.shape(Qm_sigma))
            print("shape of the Qm's after val: ", np.shape(Qm_sigma[valid_idx_total]))
            print("shape of the Qm's after val trimmed: ", np.shape(Qm_sigma[valid_idx_total][trim_left:trim_right]))
            print("shape of the histogram slices: ", np.shape(np.append(x_slice[trim_left:trim_right],
                                          y_slice[trim_left:trim_right])))
            res = spo.curve_fit(xy_eta_f, Qm_sigma[valid_idx_total][trim_left:trim_right],
                                np.append(x_slice[trim_left:trim_right],
                                          y_slice[trim_left:trim_right]),
                                sigma=np.append(sigmas[trim_left:trim_right],
                                                sigmas2[trim_left:trim_right]), p0=[0.5, 0.5, 0, np.pi/2, 0])

            print('Unconstrained')
            print('Fit Imbar/sigma: ' + str(res[0][0]) + ', +- ' + str(np.sqrt(res[1])[0, 0]))
            print('Fit Quantum Efficiency: ' + str(res[0][1])+', +- '+ str(np.sqrt(res[1])[1,1]))
            print('Fit thetaX: ' + str(res[0][2]), ', +- ' + str(np.sqrt(res[1])[2, 2]))
            print('Fit thetaY: ' + str(res[0][3]), ', +- ' + str(np.sqrt(res[1])[3, 3]))
            print('Fit offset: ' + str(res[0][4]), ', +- ' + str(np.sqrt(res[1])[4, 4]))
            print(' ')
            print('Theta difference (degrees): ',(res[0][2]-res[0][3])*360/2/np.pi)

            n_samples_x = den_hist[n//2, valid_idx_total][trim_left:trim_right]
            n_samples_y = den_hist2[n2//2, valid_idx_total][trim_left:trim_right]

            data_x = x_slice[trim_left:trim_right]
            data_y = y_slice[trim_left:trim_right]

            # x_uncertainties_low = data_x*(1-1/2/np.sqrt(n_samples_x))
            # x_uncertainties_high = data_x*(1+1/2/np.sqrt(n_samples_x))
            x_uncertainties_low = data_x - sigmas_plot[trim_left:trim_right]
            x_uncertainties_high = data_x + sigmas_plot[trim_left:trim_right]

            bardata_x = np.array([x_uncertainties_low, x_uncertainties_high])

            # y_uncertainties_low = data_y*(1-1/2/np.sqrt(n_samples_y))
            # y_uncertainties_high = data_y*(1+1/2/np.sqrt(n_samples_y))
            y_uncertainties_low = data_y - sigmas2_plot[trim_left:trim_right]
            y_uncertainties_high = data_y + sigmas2_plot[trim_left:trim_right]

            print(x_uncertainties_low)
            print(y_uncertainties_low)

            bardata_y = np.array([y_uncertainties_low, y_uncertainties_high])

            print("DEBUG: ", np.shape(Qm_sigma[valid_idx2]))
            print("DEBUG: ", np.shape(bardata_y))


            if plot:
                x_ax = Qm_sigma[valid_idx_total][trim_left:trim_right]

                ax.plot(x_ax, data_x, '.', label=r'$\langle X \rangle_c$', color = gain_colors[1], bardata = bardata_x, barcolor = gain_colors[1])

                ax.plot(x_ax,
                         x_eta_f(x_ax, res[0][0], res[0][1], res[0][2], res[0][4]),
                        color = gain_colors[1])

                x_ax2 = Qm_sigma[valid_idx_total][trim_left:trim_right]
                ax.plot(x_ax2, data_y, '.', label=r'$\langle Y \rangle_c$', color = gain_colors[2], bardata = bardata_y, barcolor = gain_colors[2])

                ax.plot(x_ax2,
                        x_eta_f(x_ax2, res[0][0], res[0][1], res[0][3], res[0][4]),
                        color = gain_colors[2])

                Im_sigma = x / sigma[amp_i]
                Qm_sigma = y / sigma[amp_i]
                # ax.plot(Qm_sigma[valid_idx], den_hist[50][valid_idx])
                if plot_samples:
                    samples_ax = ax.panel('b')
                    samples_ax.bar(x_ax, num_hist[valid_idx[0], 50]/np.max(num_hist), color = gain_colors[0])
                    print('\n\n\ntotal_fit_samples: \n\n\n',n_samples_x)
                else:
                    samples_ax = None

    return res, ax, samples_ax

def pepsi_plot(num_hists, den_hists, ampArray, x, y, initState=1, cutoff=100):

    # gs = pplt.GridSpec(nrows = len(ampArray), ncols = 3, pad = 0, wspace = 0.2, hspace = 0.2, sharex = True, sharey = True)
    # fig = pplt.figure(span=False, refwidth=1, share = False)
    fig, gs = pplt.subplots(nrows = 2, ncols = 3,
                            axwidth = "0.75in",
                            xtickdir = 'in', ytickdir = 'in',
                            wspace = 0.3, hspace = 0.3,
                            xlocator = 8, ylocator = 8,
                            tickminor = False)
    # gs.format()
    # print("sciprot_axs_shape ", np.shape(axs))
    axisArray = ['X', 'Y', 'Z']

    tomoAxisArray = [0, 1, 2]

    sum_den_hists = np.sum(den_hists, axis=0)

    for tomoAxis in tomoAxisArray:
        for amp_i in range(0, 2):
            # ax0 = fig.subplot(gs[amp_i, 0])
            ax = gs[amp_i, tomoAxis]
            #axs[amp_i, tomoAxis + 1]
            if amp_i == 0:
                num_hist = num_hists[tomoAxis, initState, 0, :, :]
                den_hist = den_hists[tomoAxis, initState, 0, :, :]
            else:
                num_hist = num_hists[tomoAxis, initState, 3, :, :]
                den_hist = den_hists[tomoAxis, initState, 3, :, :]

            img = hist2img(num_hist, den_hist, cutoff=cutoff)

            im = ax.imshow(img.transpose(), extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            # im = ax.pcolormesh(x, y, img, extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
            im_cb = im
            # ax.set_xticks([])
            # ax.set_yticks([])

            if tomoAxis == 0:
                pass
                # ax.format(ylabel = str(np.round(ampArray[amp_i]*1000, 0))+' mV')
            # ax0.set_xticks([])
            # ax0.set_yticks([])
            if amp_i == 0:
                ax.set_title(axisArray[tomoAxis])
                # ax.set_ylabel("")

    # cbar = fig.colorbar(plt.get_cmap('seismic'), length = 0.25*1.5, ticks = [0,1], location = 'right', title = r"$\langle X \rangle_c, \langle Y \rangle_c, \langle Z \rangle_c $ respectively")
    # cbar.ax.set_yticklabels([-1, 1])
    return fig

#%%
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import os
    cname = os.environ['COMPUTERNAME']
    if cname == 'DESKTOP-CFTSB1E':
        fp = r"E:/Ryan_Files/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/"
        save_hist_fp = r'E:/Ryan_Files\OneDrive - University of Pittsburgh\slides_figures\science_protocol.svg'
        save_fit_fp = r'E:/Ryan_Files\OneDrive - University of Pittsburgh\slides_figures\science_protocol_fit.svg'
    else:
        fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/'
        save_hist_fp = r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\science_protocol.svg'
        save_fit_fp = r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\science_protocol_fit.svg'
    # fp = r"E:/Ryan_Files/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/"
    # fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_lower_pwrs_more_recs/science_protocol_combined_hists.h5'
    # fp = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/paper_data/NISTAMP_2022/science_protocol_multi_rep/science_protocol_multiRep/science_protocol_multiRep_hists.h5'
    name = r'science_protocol_combined'
    # mpl.use('Qt5Agg')
    res = load_hist(fp, name, aligned=True)
    num_hists, den_hists, ampArray, x, y, Imbar, Qmbar, sigma = res

    cutoff = 10
    fig = pepsi_plot(num_hists, den_hists, ampArray, x/np.average(sigma), y/np.average(sigma), cutoff = cutoff)
    fig.save(save_hist_fp)
    tomo_axis = 0
    init_state = 1
    amp_i = 3
    direction = 1
    #linecut_fit(num_hists, den_hists, x, y, Imbar, Qmbar, sigma, args, cutoff=1, plot=True, constrained=False)
    # plt.rcParams()
    trim_left = 4
    trim_right = -4
    refaspect = 3
    fitfig, fitax = pplt.subplots(refwidth = '3in', refaspect = refaspect)
    res, lc_fit_ax, samples_ax = linecut_fit(num_hists,

                                 den_hists,
                                 x, y,
                                 Imbar, Qmbar,
                                 sigma,
                                 [tomo_axis, init_state, amp_i, direction],
                                 constrained = False,
                                 ax = fitax,
                                 plot_samples = False,
                                 trim_left = trim_left,
                                 trim_right = trim_right,
                                 cutoff = cutoff)

    # ticklabelpad = mpl.rcParams['ytick.major.pad']
    # lc_fit_ax.set_ylim(-0.6, 0.6)
    # lc_fit_ax.set_xlabel("")

    # samples_ax.annotate(r"$\overline{Q_m}$", xy=(1,0), xytext=(5, 0), ha='left', va='top',
    #             xycoords='axes fraction', textcoords='offset points')
    # lc_fit_ax.annotate(r"$\langle X \rangle_c$"+"\n\n"+r"$\langle Y \rangle_c$", xy=(-0.175,0.45), xytext=(-ticklabelpad, 5), ha='left', va='top',
                # xycoords='axes fraction', textcoords='offset points')

    # lc_fit_ax.set_ylabel(r"$\langle X \rangle$")
    # lc_fit_ax.set_yticks([-0.5, 0, 0.5])
    # lc_fit_ax.set_yticklabels([-0.5, 0, 0.5])
    # lc_fit_ax.legend(location = 'top', ncols = 2, frame = 0, markersize = 10)

    if samples_ax is not None:
        pass
        # samples_ax.set_xlabel(r"$Q_m}$")
        # samples_ax.set_ylabel('Samples')
        # samples_ax.yaxis.tick_right()
    else:
        pass
        # lc_fit_ax.set_xlabel(r"$Q_m/\sigma$")

    plt.show()
fitfig.save(save_fit_fp)
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
