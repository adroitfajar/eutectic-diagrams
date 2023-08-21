# -*- coding: utf-8 -*-
"""
This script is to compare prediction results using visual images

Melting Temperature
Data source:
    (1) 'predicted-mp_rf.csv' ===>> prediction of melting temperatures using random forest
    (2) 'predicted-mp_xgb.csv' ===>> prediction of melting temperatures using xgboost
    (2) 'predicted-mp_mlp.csv' ===>> prediction of melting temperatures using seq-mlp

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

sns.set_style('ticks')

### Load the dataset
dataset = pd.read_csv('predicted_mp.csv', delimiter=',', keep_default_na=False, na_values=[''])

print('\t')
print(f'Filetype: {type(dataset)}, Shape: {dataset.shape}')
print(dataset.head(n=10))

### Visualize the dataset
### Set fontsize
fontsize = 16

### Set canvas
fig = plt.figure(figsize=(8, 5), dpi=100)
ax = fig.add_subplot(111)

### Set border width and color
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
    
### Plot data
scatter = sns.scatterplot(
    data=dataset,
    x='compound',
    y='mpK',
    s=170,
    hue='algorithm',
    palette=['green', 'orange', 'purple'],
    alpha=0.5,
    ax=ax
    )

### Set interval of the canvas
interval = 10
ax.set_xlim(min(dataset['compound']) - 0.2 * interval, max(dataset['compound']) + 0.2 * interval)
ax.set_ylim(min(dataset['mpK']) - 5 * interval, max(dataset['mpK']) + 5 * interval)

### Set font properties
sizes = [fontsize, 0.9 * fontsize]
fonts = [fm.FontProperties(family='arial', size=sizes[i], weight='normal', style='normal') for i in range(len(sizes))] 

### Set axes labels
ax.set_xlabel('Compound', labelpad=10, fontproperties=fonts[0])
ax.set_ylabel('Predicted Tm (K)', labelpad=10, fontproperties=fonts[0])

### Set ticker position
ticker_arg = [5, 10, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]

ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])

### Modify fontsize of the ticklabels
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()

[(i.label.set_fontproperties('arial'), i.label.set_fontsize(sizes[0])) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(sizes[0])) for j in ycoord]

### Set legend
## get the handles and labels from scatter
handles, labels = scatter.get_legend_handles_labels()

## remove original legend
ax.get_legend().remove()

## create new legend with custom marker sizes
legend_labels = labels[0:]

from matplotlib import colors

# define colors with alpha channel
color = ['green', 'orange', 'purple']  
rgba_colors = [colors.to_rgba(c, alpha=0.5) for c in color]

custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=13) 
                 for c in rgba_colors]

ax.legend(handles=custom_legend, title='ML Model', title_fontsize='14', labels=legend_labels, 
           loc='upper left', fontsize='14', handletextpad=0.5)

fig.savefig('fig_1g.jpg', dpi=500, bbox_inches='tight')
plt.show()
