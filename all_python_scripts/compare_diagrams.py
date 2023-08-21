# -*- coding: utf-8 -*-
"""
This script is for comparing ideal vs real SLE phase diagrams

"""

### import libraries
import pandas as pd
import numpy as np

### define the function to calculate melting curves (T) in SLE phase diagrams
def sle_equation_ideal(melting, fusion):
    """
    The function to calculate curve shapes in SLE phase diagram.
        T = 1 / (1/Tm - (ln(xi*gi) * (R/DfusH)))
        All variables are iterated in parallel.
    """
    xi = np.linspace(0.1, 1, 10)
    gi = 1 # ideal mixture
    R = 8.3145
    T_curves_ideal = []
    for i, j in zip(melting, fusion):
        T = 1 / (1/i - ((np.log(xi*gi) * (R/j))))
        T_curves_ideal.append(T)
    print("\n", "Result Overview : \n", "\n",
          "at index = 0 \n",
          "\t Melting Temperature (K) =", melting[0], "\n",
          "\t Fusion Enthalpy (J mol-1) =", fusion[0], "\n",
          "\t Curve (K) =\n", T_curves_ideal[0], "\n",
          "\n", "at index = 1 \n",
          "\t Melting Temperature (K) =", melting[1], "\n",
          "\t Fusion Enthalpy (J mol-1) =", fusion[1], "\n",
          "\t Curve (K) =\n", T_curves_ideal[1],
          )
    return T_curves_ideal

### define the function to calculate melting curves (T) in SLE phase diagrams
def sle_equation_real(melting, fusion, ratio, gamma):
    """
    The function to calculate curve shapes in SLE phase diagram.
        T = 1 / (1/Tm - ((ln(xi) + ln(gi)) * (R/DfusH)))
        All variables are iterated in parallel.
    """
    R = 8.3145
    T_curves_real = []
    for i, j, k, l in zip(melting, fusion, ratio, gamma):
        T = 1 / (1/i - ((k + l) * R/j))
        T_curves_real.append(T)
    print("\n", "Result Overview : \n", "\n",
          "at index = 0 \n",
          "\t Melting Temperature (K) =", melting[0], "\n",
          "\t Fusion Enthalpy (J mol-1) =", fusion[0], "\n",
          "\t Curve (K) =\n", T_curves_real[0], "\n",
          "\n", "at index = 1 \n",
          "\t Melting Temperature (K) =", melting[1], "\n",
          "\t Fusion Enthalpy (J mol-1) =", fusion[1], "\n",
          "\t Curve (K) =\n", T_curves_real[1],
          )
    return T_curves_real

### load dataset <the results of ML predictions>
hba = pd.read_csv('input_hba.csv') # 60 chemicals
hbd = pd.read_csv('input_hbd.csv') # 50 chemicals

### define values for HBA
hba_melting = hba['mpK'] # unit = K
hba_fus_ori = hba['fusH'] # unit = kJ mol-1
hba_fusion = hba_fus_ori * 1000 # unit = J mol-1
## repeat the value of each item in hba 50 times
hba_melting_all = [item for item in hba_melting for _ in range(50)]
hba_fusion_all = [item for item in hba_fusion for _ in range(50)]

### define values for HBD
hbd_melting = hbd['mpK'] # unit = K
hbd_DfusH_ori = hbd['fusH'] # unit = kJ mol-1
hbd_fusion = hbd_DfusH_ori * 1000 # unit = J mol-1
## repeat the value of hbd list 60 times
hbd_melting_all = hbd_melting.tolist() * 60
hbd_fusion_all = hbd_fusion.tolist() * 60

### calculate curves for i in HBA
hba_curves_ideal = sle_equation_ideal(hba_melting, hba_fusion)

### calculate curves for i in HBD
hbd_curves_ideal = sle_equation_ideal(hbd_melting, hbd_fusion)

### load dataset <the results of DFT and COSMO-RS calculations>
## the value of gamma has been converted into its natural logarithmic scale (ln_gamma)
hba_gamma = np.load('hba_gamma.npy', allow_pickle=True)
hbd_gamma = np.load('hbd_gamma.npy', allow_pickle=True)

### define the ratios of the mixtures
## ratios of HBA
hba_xi = np.linspace(0.1, 1, 10)
ln_hba_xi = np.log(hba_xi)
hba_ratios = [ln_hba_xi] * len(hba_gamma)
## ratios of HBD
hbd_xi = np.linspace(1, 0.1, 10)
ln_hbd_xi = np.log(hbd_xi)
hbd_ratios = [ln_hbd_xi] * len(hbd_gamma)

### estimate curves for i in HBA
hba_curves_real = sle_equation_real(hba_melting_all, hba_fusion_all, hba_ratios, hba_gamma)

### estimate curves for i in HBA
hbd_curves_real = sle_equation_real(hbd_melting_all, hbd_fusion_all, hbd_ratios, hbd_gamma)


# ####### STORE THE DATA FOR ANALYSIS #######
# ##### IDEAL MIXTURES
# ### define X and Y
# xii = np.linspace(0.1, 1, 10)
# xji = np.linspace(0, 0.9, 10)
# xhba_ideal = [xii] * 3000
# xhbd_ideal = [xji] * 3000
# yhba_ideal = [item for item in hba_curves_ideal for _ in range(50)]
# yhbd_ideal = hbd_curves_ideal * 60
# yhbd_ideal_rev = np.array([row[::-1] for row in yhbd_ideal]) # reverse the value from big to small

# ### store the data as numpy objects
# np.save('xhba_ideal.npy', xhba_ideal)
# np.save('xhbd_ideal.npy', xhbd_ideal)
# np.save('yhba_ideal.npy', yhba_ideal)
# np.save('yhbd_ideal.npy', yhbd_ideal_rev)

# ### reload to confirm
# xa_ideal = np.load('xhba_ideal.npy', allow_pickle=True)
# xb_ideal = np.load('xhbd_ideal.npy', allow_pickle=True)
# ya_ideal = np.load('yhba_ideal.npy', allow_pickle=True)
# yb_ideal = np.load('yhbd_ideal.npy', allow_pickle=True)

# ##### REAL MIXTURES
# ### define X and Y
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# xhba_real = [xir] * 3000
# xhbd_real = [xjr] * 3000
# yhba_real = hba_curves_real
# yhbd_real = hbd_curves_real

# ### store the data as numpy objects
# np.save('xhba_real.npy', xhba_real)
# np.save('xhbd_real.npy', xhbd_real)
# np.save('yhba_real.npy', yhba_real)
# np.save('yhbd_real.npy', yhbd_real)

# ### reload to confirm
# xa_real = np.load('xhba_real.npy', allow_pickle=True)
# xb_real = np.load('xhbd_real.npy', allow_pickle=True)
# ya_real = np.load('yhba_real.npy', allow_pickle=True)
# yb_real = np.load('yhbd_real.npy', allow_pickle=True)

####### TERMINATE #######


##### SLE PHASE DIAGRAMS FOR THE MIXTURES
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

### for i at a given index -- this for hba_curves[0] >< hbd_curves[0]
composition = len(hba_curves_ideal[0])
xii = np.linspace(0.1, 1, composition)
composition = len(hbd_curves_ideal[0])
xji = np.linspace(0.9, 0, composition)
xir = np.linspace(0.1, 1, 10)
xjr = np.linspace(0, 0.9, 10)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.plot(xii, hba_curves_ideal[0], 'r--', label='HBA Ideal', lw=3, alpha=0.3)
ax.plot(xji, hbd_curves_ideal[0], 'b--', label='HBD Ideal', lw=3, alpha=0.3)
ax.plot(xir, hba_curves_real[0][0], 'r', label='HBA Real', lw=5, alpha=0.5)
ax.plot(xjr, hbd_curves_real[0][0], 'b', label='HBD Real', lw=5, alpha=0.5)
x_temp = np.linspace(0, 1, 11)
room_temp = np.full(11, 298)
ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
ticker_arg = [0.1, 0.2, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
plt.axis([0, 1, 0, 600])
[x.label1.set_fontfamily('arial') for x in xcoord]
[x.label1.set_fontsize(16) for x in xcoord]
[y.label1.set_fontfamily('arial') for y in ycoord]
[y.label1.set_fontsize(16) for y in ycoord]
ax.legend(loc='best', fontsize=14, ncol=1)
dpi_assign = 500
plt.savefig('mixtures_0-0.jpg', dpi=dpi_assign, bbox_inches='tight')


# ####### FOR MANUSCRIPT #######
# ### figure 3d -- the index on the ax.plot must be defined!
# ## MIXTURE = 2996 -- HBA = 59 HBD = 46
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[59], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[46], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2996][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2996][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_3d.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure 3e -- the index on the ax.plot must be defined!
# ## MIXTURE = 2650 -- HBA = 53 HBD = 0
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[53], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[0], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2650][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2650][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_3e.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure 3f -- the index on the ax.plot must be defined!
# ## MIXTURE = 171 -- HBA = 3 HBD = 21
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[3], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[21], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[171][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[171][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_3f.jpg', dpi=dpi_assign, bbox_inches='tight')


# ####### FOR SUPPLEMENTARY #######
# ### figure s4a -- the index on the ax.plot must be defined!
# ## MIXTURE = 1015 -- HBA = 20 HBD = 15
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[20], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[15], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[1015][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[1015][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4a.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4b -- the index on the ax.plot must be defined!
# ## MIXTURE = 1065 -- HBA = 21 HBD = 15
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[21], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[15], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[1065][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[1065][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4b.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4c -- the index on the ax.plot must be defined!
# ## MIXTURE = 2993 -- HBA = 59 HBD = 43
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[59], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[43], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2993][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2993][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4c.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4d -- the index on the ax.plot must be defined!
# ## MIXTURE = 2964 -- HBA = 59 HBD = 14
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[59], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[14], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2964][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2964][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4d.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4e -- the index on the ax.plot must be defined!
# ## MIXTURE = 2967 -- HBA = 59 HBD = 17
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[59], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[17], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2967][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2967][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4e.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4f -- the index on the ax.plot must be defined!
# ## MIXTURE = 2969 -- HBA = 59 HBD = 19
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[59], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[19], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2969][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2969][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4f.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4g -- the index on the ax.plot must be defined!
# ## MIXTURE = 0 -- HBA = 0 HBD = 0
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[0], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[0], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[0][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[0][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4g.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4h -- the index on the ax.plot must be defined!
# ## MIXTURE = 1 -- HBA = 0 HBD = 1
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[0], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[1], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[1][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[1][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4h.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4i -- the index on the ax.plot must be defined!
# ## MIXTURE = 2 -- HBA = 0 HBD = 2
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[0], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[2], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[2][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[2][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4i.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4j -- the index on the ax.plot must be defined!
# ## MIXTURE = 100 -- HBA = 2 HBD = 0
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[2], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[0], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[100][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[100][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4j.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4k -- the index on the ax.plot must be defined!
# ## MIXTURE = 318 -- HBA = 6 HBD = 18
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[6], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[18], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[318][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[318][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4k.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### figure s4l -- the index on the ax.plot must be defined!
# ## MIXTURE = 1012 -- HBA = 20 HBD = 12
# composition = len(hba_curves_ideal[0])
# xii = np.linspace(0.1, 1, composition)
# composition = len(hbd_curves_ideal[0])
# xji = np.linspace(0.9, 0, composition)
# xir = np.linspace(0.1, 1, 10)
# xjr = np.linspace(0, 0.9, 10)
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.plot(xii, hba_curves_ideal[20], 'r--', label='HBA Ideal', lw=3, alpha=0.3) # consider hba index
# ax.plot(xji, hbd_curves_ideal[12], 'b--', label='HBD Ideal', lw=3, alpha=0.3) # consider hbd index
# ax.plot(xir, hba_curves_real[1012][0], 'r', label='HBA Real', lw=5, alpha=0.5) # change the first index [this][0]
# ax.plot(xjr, hbd_curves_real[1012][0], 'b', label='HBD Real', lw=5, alpha=0.5) # for real; the index hba = hbd
# x_temp = np.linspace(0, 1, 11)
# room_temp = np.full(11, 298)
# ax.plot(x_temp, room_temp, 'k--', lw=3, alpha=0.3)
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.xlabel(r'$x_{HBA}$', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'T (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [0.1, 0.2, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([0, 1, 0, 600])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# # ax.legend(loc='best', fontsize=14, ncol=1)
# dpi_assign = 500
# plt.savefig('fig_s4l.jpg', dpi=dpi_assign, bbox_inches='tight')