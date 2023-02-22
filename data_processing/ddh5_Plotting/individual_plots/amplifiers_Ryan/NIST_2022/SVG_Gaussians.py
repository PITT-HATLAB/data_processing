# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:44:29 2023

make a bunch of nice, scalable gaussians that you can drag and drop into PPTX
or Illustrator or whatever you like

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
import warnings

plt.style.use('hatlab')
cmap = mpl.cm.get_cmap('Seismic')
from_edge = 0
cnum = np.linspace(0+from_edge, 1-from_edge, 10)
newcolors = cmap(cnum)

greens = mpl.cm.get_cmap('greens')
greencmap = ListedColormap(greens(cnum))
blues = mpl.cm.get_cmap('blues')
bluecmap = ListedColormap(blues(cnum))
reds = mpl.cm.get_cmap('reds')
redcmap = ListedColormap(reds(cnum))

# pink = np.array([248/256, 24/256, 148/256, 1])
# newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)
# newcmp = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
from cycler import cycler
default_prop_cycler = cycler('color', [newcmp(cnum[0]), newcmp(cnum[1]), newcmp(cnum[2])])
plot_colors = [redcmap(cnum[5]), greencmap(0.5), bluecmap(cnum[5])]

plt.style.use('hatlab')

gaussian = lambda x, s: np.exp(-(x**2)/2/s**2)
x = np.linspace(-1, 1)

names = ['red', 'green', 'blue']
for plot_color, name in zip(plot_colors, names): 
    fig, ax = pplt.subplots(facecolor = None)
    ax.axis('off')
    s = 0.35
    ax.plot(x, gaussian(x, s), color = 'k')
    ax.fill_between(x, gaussian(x, s), color = plot_color)
    fig.savefig(f"gaussian_{name}.svg")
