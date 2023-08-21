# -*- coding: utf-8 -*-
"""
This script executes estimation of SLE phase diagrams (REAL) for any given DES*.
The curve in an SLE phase diagram follows this equation:
    ln(xigi) = (DfusH/R)((1/Tm)-(1/T))
Where,
    xi    : certain liquid mole fraction composition of pure compound i
    gi    : activity coeficient of compound i (for a real mixture, g < or > 1)
    DfusH : fusion enthalpy of the pure compound (J mol-1)
    Tm    : melting temperature of the pure compound (K)
    R     : the universal gas constant (8.3145 J mol-1 K-1)
    T     : the absolute temperature of the curve (K) ==>> solve for this T

The values of DfusH and Tm are provided by ML predictions.
<< we selected RF! XGB seems to overfit, and ANN seems to underfit in this case >>
The values of gi is provided by DFT and COSMO-RS calculations (execute the file 'gamma.py')

*DES is a subclass of eutectic mixtures, typically consisting of a hydrogen bond acceptor (HBA) and a hydrogen bond donor (HBD)

Required files:
    HBA = "input_hba.csv"
    HBD = "input_hbd.csv"
    gamma = "hba_gamma.npy" and "hbd_gamma.npy"

NOTE: this script focuses on REAL MIXTURES! 

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


##### SLE PHASE DIAGRAMS FOR EUTECTIC MIXTURES
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

### for i at a given index -- this for hba_curves[0] >< hbd_curves[0]
xi = np.linspace(0.1, 1, 10)
xj = np.linspace(0, 0.9, 10)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.plot(xi, hba_curves_real[0][0], 'r', xj, hbd_curves_real[0][0], 'b', lw=5, alpha=0.5)
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
dpi_assign = 500
plt.savefig('mixture_real_0-0.jpg', dpi=dpi_assign, bbox_inches='tight')

### for i at a given index -- this for hba_curves[0] >< hbd_curves[1]
xi = np.linspace(0.1, 1, 10)
xj = np.linspace(0, 0.9, 10)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.plot(xi, hba_curves_real[1][0], 'r', xj, hbd_curves_real[1][0], 'b', lw=5, alpha=0.5)
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
dpi_assign = 500
plt.savefig('mixture_real_0-1.jpg', dpi=dpi_assign, bbox_inches='tight')


##### CREATE ANIMATIONS
import matplotlib.animation as animation

### define some constant
N_DES = len(hba_curves_real)

### define a function to plot each diagram
def plot_diagram(hba, hbd, ax):
    xi = np.linspace(0.1, 1, 10)
    xj = np.linspace(0, 0.9, 10)
    for m in ['top', 'bottom', 'left', 'right']:
        ax.spines[m].set_linewidth(1)
        ax.spines[m].set_color('black')
    ax.plot(xi, hba[0], 'r', xj, hbd[0], 'b', lw=5, alpha=0.5)
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

### define the animation function
def animate(i, fig, ax):
    # clear the previous plot
    ax.clear()
    # get the index for HBA and HBD curves based on the frame number
    i_curve = i % N_DES
    # plot the corresponding HBA and HBD curves with the same index
    plot_diagram(hba_curves_real[i_curve], hbd_curves_real[i_curve], ax)
    # add a title to the plot
    ax.set_title(f"Real Mixture {i_curve}", fontweight='bold', fontsize=12)
    return fig,

### create the figure outside of the animate function
fig, ax = plt.subplots(figsize=(7.5, 5.5))

### create the animation
ani = animation.FuncAnimation(fig, animate, frames=N_DES, fargs=(fig, ax), interval=50)

### save the animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save('animation_real.mp4', writer=writer, dpi=300)


##### SHOW ALL DIAGRAMS
### Specify the new folder name
folder_name = 'all_real_diagrams'

### Specify the parent directory where the new folder will be created
parent_directory = 'D:/Cheminformatics/eu_diagrams'

### Combine the parent directory and folder name to get the complete folder path
folder_path = os.path.join(parent_directory, folder_name)

### Create the new folder if it doesn't already exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

### define the function to save diagrams as images
def save_diagram_image(hba, hbd, index):
    xi = np.linspace(0.1, 1, 10)
    xj = np.linspace(0, 0.9, 10)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for m in ['top', 'bottom', 'left', 'right']:
        ax.spines[m].set_linewidth(1)
        ax.spines[m].set_color('black')
    ax.plot(xi, hba, 'r', xj, hbd, 'b', lw=5, alpha=0.5)
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
    ax.set_title(f"Real Mixture {index}", fontweight='bold', fontsize=12)
    dpi_assign = 300
    file_path = os.path.join(folder_path, f'mixture_real_{index}.jpg')
    plt.savefig(file_path, dpi=dpi_assign, bbox_inches='tight')
    plt.close(fig)

### save each diagram as a separate image
for i in range(len(hba_curves_real)):
    save_diagram_image(hba_curves_real[i][0], hbd_curves_real[i][0], i)