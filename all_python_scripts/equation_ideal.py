# -*- coding: utf-8 -*-
"""
This script executes estimation of SLE phase diagram (IDEAL) for any given DES*.
The curve in SLE phase diagram follows this equation:
    ln(xigi) = (DfusH/R)((1/Tm)-(1/T))
Where,
    xi    : certain liquid mole fraction composition of pure compound i
    gi    : activity coeficient of compound i (for an ideal mixture, g = 1)
    DfusH : fusion enthalpy of the pure compound (J mol-1)
    Tm    : melting temperature of the pure compound (K)
    R     : the universal gas constant (8.3145 J mol-1 K-1)
    T     : the absolute temperature of the curve (K) ==>> solve for this T

The values of DfusH and Tm are provided by ML predictions.
<< we selected RF! XGB seems to overfit, and ANN seems to underfit in this case >>

*DES is a subclass of eutectic mixtures, typically consisting of a hydrogen bond acceptor (HBA) and a hydrogen bond donor (HBD)

Required files:
    HBA = "input_hba.csv"
    HBD = "input_hbd.csv"

NOTE: this script focuses on IDEAL MIXTURES! 

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

### load dataset <the results of ML predictions>
hba = pd.read_csv('input_hba.csv') # 60 chemicals
hbd = pd.read_csv('input_hbd.csv') # 50 chemicals

### define values for HBA
hba_melting = hba['mpK'] # unit = K
hba_fus_ori = hba['fusH'] # unit = kJ mol-1
hba_fusion = hba_fus_ori * 1000 # unit = J mol-1

### define values for HBD
hbd_melting = hbd['mpK'] # unit = K
hbd_DfusH_ori = hbd['fusH'] # unit = kJ mol-1
hbd_fusion = hbd_DfusH_ori * 1000 # unit = J mol-1

### calculate curves for i in HBA
HBA_curves = sle_equation_ideal(hba_melting, hba_fusion)

### calculate curves for i in HBD
HBD_curves = sle_equation_ideal(hbd_melting, hbd_fusion)


##### SLE PHASE DIAGRAMS FOR EUTECTIC MIXTURES
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

### for i at a given index -- this for HBA_curves[0] >< HBD_curves[0]
composition = len(HBA_curves[0])
xi = np.linspace(0.1, 1, composition)
composition = len(HBD_curves[0])
xj = np.linspace(0.9, 0, composition)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.plot(xi, HBA_curves[0], 'r', xj, HBD_curves[0], 'b', lw=5, alpha=0.5)
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
plt.savefig('mixture_ideal_0-0.jpg', dpi=dpi_assign, bbox_inches='tight')

### for i at a given index -- this for HBA_curves[0] >< HBD_curves[1]
composition = len(HBA_curves[0])
xi = np.linspace(0.1, 1, composition)
composition = len(HBD_curves[0])
xj = np.linspace(0.9, 0, composition)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.plot(xi, HBA_curves[0], 'r', xj, HBD_curves[1], 'b', lw=5, alpha=0.5)
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
plt.savefig('mixture_ideal_0-1.jpg', dpi=dpi_assign, bbox_inches='tight')


##### CREATE ANIMATIONS
import matplotlib.animation as animation

### define some constants
N_HBA = len(hba_melting)
N_HBD = len(hbd_melting)
N_COMBOS = N_HBA * N_HBD

### define a function to plot each diagram
def plot_diagram(hba, hbd, ax):
    xi = np.linspace(0.1, 1, 10)
    xj = np.linspace(0.9, 0, 10)
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

### define the animation function
def animate(i, fig, ax):
    # clear the previous plot
    ax.clear()
    # get the indices of the HBA and HBD arrays corresponding to the current frame
    i_hba, i_hbd = divmod(i, N_HBD)
    # plot the corresponding diagram
    plot_diagram(HBA_curves[i_hba], HBD_curves[i_hbd], ax)
    # add a title to the plot
    ax.set_title(f"HBA {i_hba}, HBD {i_hbd}", fontweight='bold', fontsize=12)
    return fig,

### create the figure outside of the animate function
fig, ax = plt.subplots(figsize=(7.5, 5.5))

### create the animation
ani = animation.FuncAnimation(fig, animate, frames=N_COMBOS, fargs=(fig, ax), interval=50)

### save the animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
ani.save('animation_ideal.mp4', writer=writer, dpi=300)


##### SHOW ALL DIAGRAMS
### define the function to save diagrams as images
def save_diagram_image(hba, hbd, index, folder_path):
    xi = np.linspace(0.1, 1, 10)
    xj = np.linspace(0.9, 0, 10)
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
    ax.set_title(f"Ideal Mixture {index}", fontweight='bold', fontsize=12)
    dpi_assign = 300
    plt.savefig(os.path.join(folder_path, f'mixture_ideal_{index}.jpg'), dpi=dpi_assign, bbox_inches='tight')
    plt.close(fig)

### create the folder path to save the images and the animation
folder_path = 'D:/Cheminformatics/eu_diagrams/all_ideal_diagrams'

### create the folder if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

### save each diagram as a separate image
for i in range(len(HBA_curves)):
    for j in range(len(HBD_curves)):
        save_diagram_image(HBA_curves[i], HBD_curves[j], (i * N_HBD) + j, folder_path)