#importing the data to reproduce Sivak's figure
import pandas as pd
import numpy as np
import proplot as pplt
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
mpl.style.use('hatlab')

CSV = r'C:/Users/Ryan/OneDrive - University of Pittsburgh/slides_figures/Sivak_Fig_vals.csv'
df = pd.read_csv(CSV)
fig, ax = pplt.subplots(width = '14in')
mpl_markers = ['.', 'o','v','^','<','>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', '+', 'D']
prop_cycler = cycler(marker = mpl_markers)
ax.set_prop_cycle(prop_cycler)
# ax.set_aspect(0.5)
fontsize = 25

line50 = lambda x: x-3
line001 = lambda x: x-30
line0001 = lambda x: x-40

for key, device in df.T.items():
    if key == list(df.T.items())[-1][0]: 
        color = 'red'
    else: 
        color = 'black'
    ax.scatter(device['Pump Power (dBm)'], device['Output Saturation Power (dBm)'], color = color, markersize = 150, label = str(device['Device'])+' '+str(device['Source']))

xvals = np.linspace(-100, -20, 100)
ax.plot(xvals, line50(xvals), 'k-', label = '100% efficiency')
ax.plot(xvals, line001(xvals), '--', label = '0.1% efficiency')
# ax.plot(xvals, line0001(xvals), '--', label = '0.01% efficiency')
ax.legend(loc = 'right', ncols = 1)
ax.set_xlabel('Pump Power (dBm)', fontsize = fontsize)
ax.set_ylabel('Output Saturation Power (dBm)', fontsize = fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

fig.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\pumpEfficiencyFig.pdf')