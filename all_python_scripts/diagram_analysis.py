# -*- coding: utf-8 -*-
"""
Analysis of the SLE phase diagrams to locate the eutectic coordinates

"""

### import libraries
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

##### IDEAL MIXTURES
### load the data
xhba_ideal = np.load('xhba_ideal.npy', allow_pickle=True)
xhbd_ideal = np.load('xhbd_ideal.npy', allow_pickle=True)
yhba_ideal = np.load('yhba_ideal.npy', allow_pickle=True)
yhbd_ideal = np.load('yhbd_ideal.npy', allow_pickle=True)

### provide empty lists to store the results
xcross_ideal = []
ycross_ideal = []

### iterate through all pairs of curves in the lists
for i in range(len(xhba_ideal)):
    x1, y1 = xhba_ideal[i], yhba_ideal[i]
    x2, y2 = xhbd_ideal[i], yhbd_ideal[i]

    f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
    f2 = interp1d(x2, y2, kind='cubic', fill_value="extrapolate")

    def f3(x):
        return f1(x) - f2(x)

    # Using min and max of x-values to dynamically create the bracket.
    bracket_min = min(min(x1), min(x2))
    bracket_max = max(max(x1), max(x2))

    try:
        result = optimize.root_scalar(f3, method='brentq', bracket=[bracket_min,bracket_max]) 
        if result.converged:
            intersect_x = result.root
            intersect_y = f1(intersect_x)
            xcross_ideal.append(intersect_x)
            ycross_ideal.append(intersect_y)
        else:
            xcross_ideal.append(None)
            ycross_ideal.append(None)
    except ValueError:
        xcross_ideal.append(None)
        ycross_ideal.append(None)


##### REAL MIXTURES
### load the data
xa_real = np.load('xhba_real.npy', allow_pickle=True)
xb_real = np.load('xhbd_real.npy', allow_pickle=True)
ya_real = np.load('yhba_real.npy', allow_pickle=True)
yb_real = np.load('yhbd_real.npy', allow_pickle=True)

### define X and Y
xhba_real = xa_real
xhbd_real = xb_real
yhba_real = ya_real.reshape((3000, 10))
yhbd_real = yb_real.reshape((3000, 10))

### provide empty lists to store the results
xcross_real = []
ycross_real = []

### iterate through all pairs of curves in the lists
for i in range(len(xhba_real)):
    x1, y1 = xhba_real[i], yhba_real[i]
    x2, y2 = xhbd_real[i], yhbd_real[i]

    f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
    f2 = interp1d(x2, y2, kind='cubic', fill_value="extrapolate")

    def f3(x):
        return f1(x) - f2(x)

    # Using min and max of x-values to dynamically create the bracket.
    bracket_min = min(min(x1), min(x2))
    bracket_max = max(max(x1), max(x2))

    try:
        result = optimize.root_scalar(f3, method='brentq', bracket=[bracket_min,bracket_max]) 
        if result.converged:
            intersect_x = result.root
            intersect_y = f1(intersect_x)
            xcross_real.append(intersect_x)
            ycross_real.append(intersect_y)
        else:
            xcross_real.append(None)
            ycross_real.append(None)
    except ValueError:
        xcross_real.append(None)
        ycross_real.append(None)


# ##### SUPPLEMENTARY ANALYSIS
# ### load dataset <the results of ML predictions>
# import pandas as pd
# hba = pd.read_csv('input_hba.csv') # 60 chemicals
# hbd = pd.read_csv('input_hbd.csv') # 50 chemicals

# ### define values for HBA
# hba_melting = hba['mpK'] # unit = K
# hba_fus_ori = hba['fusH'] # unit = kJ mol-1
# hba_fusion = hba_fus_ori * 1 # unit = kJ mol-1
# ## repeat the value of each item in hba 50 times
# hba_melting_all = [item for item in hba_melting for _ in range(50)]
# hba_fusion_all = [item for item in hba_fusion for _ in range(50)]

# ### define values for HBD
# hbd_melting = hbd['mpK'] # unit = K
# hbd_DfusH_ori = hbd['fusH'] # unit = kJ mol-1
# hbd_fusion = hbd_DfusH_ori * 1 # unit = kJ mol-1
# ## repeat the value of hbd list 60 times
# hbd_melting_all = hbd_melting.tolist() * 60
# hbd_fusion_all = hbd_fusion.tolist() * 60

# ### load dataset <the results of DFT and COSMO-RS calculations>
# ## the value of gamma has been converted into its natural logarithmic scale (ln_gamma)
# hba_gamma = np.load('hba_gamma.npy', allow_pickle=True)
# hbd_gamma = np.load('hbd_gamma.npy', allow_pickle=True)


# ##### VISUALIZATIONS
# ### import libraries
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import matplotlib.ticker as mticker
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')

# ### EUTECTIC MOLE FRACTIONS at the intersection points
# mixture = np.linspace(1, 3000, 3000)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(mixture, xcross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(mixture, xcross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'Mixture', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$x_{HBA}$ = $x_E$', labelpad=10, fontproperties=fonts)
# ticker_arg = [375, 750, 0.1, 0.2]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-30, 3030, -0.02, 1.02])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='lower right', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_3a.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### EUTECTIC TEMPERATURES at the intersection points
# mixture = np.linspace(1, 3000, 3000)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(mixture, ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(mixture, ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'Mixture', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [375, 750, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-30, 3030, 100, 500])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='lower right', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_3b.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### EUTECTIC COORDINATES at the intersection points
# mixture = np.linspace(1, 3000, 3000)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(xcross_ideal, ycross_ideal, 50, 'black', alpha=0.2, label='Ideal')
# ax.scatter(xcross_real, ycross_real, 50, 'green', alpha=0.2, label='Real')
# plt.xlabel(r'$x_{HBA}$ = $x_E$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-0.02, 1.02, 100, 500])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# ax.legend(loc='upper right', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_3c.jpg', dpi=dpi_assign, bbox_inches='tight')


# ##### SUPPLEMENTARY FIGURES
# ### HBA Tm vs TE
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hba_melting_all, ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hba_melting_all, ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'$T_{m,HBA}$ (K)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [12.5, 25, 12.5, 25]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([255, 350, 220, 350])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3a.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### HBD Tm vs TE
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hbd_melting_all, ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hbd_melting_all, ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'$T_{m,HBD}$ (K)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [25, 50, 37.5, 75]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([305, 510, 145, 500])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3b.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### HBA fus vs TE
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hba_fusion_all, ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hba_fusion_all, ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'$\Delta_{fus}H_{HBA}$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [10, 20, 25, 50]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([5, 70, 150, 350])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3c.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### HBD fus vs TE
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hbd_fusion_all, ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hbd_fusion_all, ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'$\Delta_{fus}H_{HBD}$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [10, 20, 25, 50]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([10, 60, 145, 400])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3d.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### HBA ln_gamma vs TE
# hba_data = hba_gamma.reshape((3000, 10))
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hba_data[:, 5], ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hba_data[:, 5], ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'ln $\gamma_{HBA}$ ($-$)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.5, 1, 25, 50]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-3, 1.5, 150, 350])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3e.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### HBD ln_gamma vs TE
# hbd_data = hbd_gamma.reshape((3000, 10))
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(hbd_data[:, 6], ycross_ideal, 50, 'black', alpha=0.1, label='Ideal')
# ax.scatter(hbd_data[:, 6], ycross_real, 50, 'green', alpha=0.1, label='Real')
# plt.xlabel(r'ln $\gamma_{HBD}$ ($-$)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'$T_E$ (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [1, 2, 37.5, 75]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-7, 4, 145, 500])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='upper left', fontsize=14, ncol=1, handlelength=1, handletextpad=0.5)
# dpi_assign = 500
# plt.savefig('fig_s3f.jpg', dpi=dpi_assign, bbox_inches='tight')