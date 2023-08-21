# -*- coding: utf-8 -*-
"""
Visulaizations for the calculated gamma

"""

### import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.legend_handler import HandlerTuple

### load the datasets (numpy binary objects)
hba_gamma = np.load('hba_gamma.npy', allow_pickle=True)
hbd_gamma = np.load('hbd_gamma.npy', allow_pickle=True)

### Reshape data to be 2D (3000, 10)
hba_data = hba_gamma.reshape((3000, 10))
hbd_data = hbd_gamma.reshape((3000, 10))

### create a new colormap that includes an alpha channel
cmap = cm.get_cmap('rainbow_r', 10)
alpha = 0.7  # Set the transparency level
colors = cmap(np.linspace(0, 1, 10))
colors[:, -1] = alpha  # Set the alpha channel of the colors
cmap_alpha = cm.colors.ListedColormap(colors)

### plot the figure -- HBA
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
for i in range(10):
    ax.plot(hba_data[:, i], color=cmap_alpha(i), label=r'$\it{x}_{HBA}$='+str(round(0.1 * (i + 1), 1)))
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.xlabel(r'HBA in Mixture', labelpad=10, fontproperties=fonts)
plt.ylabel(r'ln $\gamma_{i}$ ($-$)', labelpad=10, fontproperties=fonts)
ticker_arg = [250, 500, 2.5, 5]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
ax.legend(loc='lower right', fontsize=12, ncol=2)
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
plt.axis([-50, 3050, -17, 8])
[x.label1.set_fontfamily('arial') for x in xcoord]
[x.label1.set_fontsize(16) for x in xcoord]
[y.label1.set_fontfamily('arial') for y in ycoord]
[y.label1.set_fontsize(16) for y in ycoord]
dpi_assign = 500
plt.savefig('fig_2b.jpg', dpi=dpi_assign, bbox_inches='tight')


### plot the figure -- HBD
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
for i in range(9, -1, -1):
    ax.plot(hbd_data[:, i], color=cmap_alpha(i), label=r'$\it{x}_{HBA}$='+str(round(0.1 * i, 1)))
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.xlabel(r'HBD in Mixture', labelpad=10, fontproperties=fonts)
plt.ylabel(r'ln $\gamma_{i}$ ($-$)', labelpad=10, fontproperties=fonts)
ticker_arg = [250, 500, 2.5, 5]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
ax.legend(loc='lower right', fontsize=12, ncol=2, handler_map={tuple: HandlerTuple(ndivide=None, pad=1.5)})
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
plt.axis([-50, 3050, -17, 8])
[x.label1.set_fontfamily('arial') for x in xcoord]
[x.label1.set_fontsize(16) for x in xcoord]
[y.label1.set_fontfamily('arial') for y in ycoord]
[y.label1.set_fontsize(16) for y in ycoord]
dpi_assign = 500
plt.savefig('fig_2d.jpg', dpi=dpi_assign, bbox_inches='tight')