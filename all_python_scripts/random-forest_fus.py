# -*- coding: utf-8 -*-
"""
Machine Learning Regression Model -- RF

This script executes learning and prediction of fusion enthalpy using Random Forest
Dataset: CRC Handbook of Chemistry and Physics 95th Edition ("fus_ent_data.csv" ---RDkit--> "fus_ent_desc.csv")
         <Please run the file named "descriptor_fus.py" before executing the present script>
Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn
Learning Algorithm: Random Forest Regressor

Author: 
    Adroit T.N. Fajar, Ph.D. (Dr.Eng.)
    Scopus Author ID: 57192386143
    ResearcherID: HNI-7382-2023
    ResearchGate: https://researchgate.net/profile/Adroit-Fajar
    GitHub: https://github.com/adroitfajar

"""

### configure the number of available CPU
import os as os
cpu_number = os.cpu_count()
n_jobs = cpu_number - 2

### import some standard libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

### load and define dataframe for the learning dataset
Learning = pd.read_csv("fus_ent_desc.csv") # this contains >500 data points

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### define X and Y out of the learning data (X: features, Y: values)
X = Learning.drop('fusH', axis=1)
Y = Learning['fusH']

print('\n')
print('Features (X): \n')
print(f'Filetype: {type(X)}, Shape: {X.shape}')
print(X)

print('\n')
print('Label (Y): \n')
print(f'Filetype: {type(Y)}, Shape: {Y.shape}')
print(Y)

### split the learning data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

### train and evaluate the model
from sklearn.ensemble import RandomForestRegressor
RFrgs = RandomForestRegressor(random_state=1)
from sklearn.model_selection import cross_val_score
cross_val = 5
scores = cross_val_score(RFrgs, X_train, Y_train, scoring="r2", cv=cross_val, n_jobs=n_jobs)
def display_score(scores):
    print('\n')
    print('Preliminary run: \n')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_score(scores)

### fine tune the model using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [100, 300, 500],
      'max_depth': [10, 50, 100],
      'max_features': [13, 26, 52, 104, 208]}
    ]
grid_search = GridSearchCV(RFrgs, param_grid, scoring="r2", cv=cross_val, n_jobs=n_jobs)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

cvres = grid_search.cv_results_

print('\n')
print('Hyperparameter tuning: \n')
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
    
grid_search.best_estimator_

### re-train the model with the best hyperparameters and the whole training set
RFrgs_opt = grid_search.best_estimator_
model = RFrgs_opt.fit(X_train, Y_train)

### analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(RFrgs_opt, X_train, Y_train, cv=cross_val, n_jobs=n_jobs)
R2_cv = r2_score(Y_train, cv_pred)
RMSE_cv = np.sqrt(mean_squared_error(Y_train, cv_pred))

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_cv)
print('RMSE score', RMSE_cv)

### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, cv_pred, 70, 'tab:blue', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel(r'Measured $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
plt.ylabel(r'Predicted $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
import matplotlib.ticker as mticker
ticker_arg = [10, 20, 10, 20]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(16)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(16)) for j in ycoord]
dpi_assign = 500
plt.savefig('rf_fus_fig1.jpg', dpi=dpi_assign, bbox_inches='tight')

### analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)
R2_train = r2_score(Y_train, train_pred)
RMSE_train = np.sqrt(mean_squared_error(Y_train, train_pred))

print('\n')
print('Learning results for training set (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_train)
print('RMSE score', RMSE_train)

### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, train_pred, 70, 'tab:grey', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel(r'Measured $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
plt.ylabel(r'Predicted $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
ticker_arg = [10, 20, 10, 20]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(16)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(16)) for j in ycoord]
dpi_assign = 500
plt.savefig('rf_fus_fig2.jpg', dpi=dpi_assign, bbox_inches='tight')

### analyze and visualize the optimized model performance on TEST SET via a SINGLE RUN
test_pred = model.predict(X_test)
R2_test = r2_score(Y_test, test_pred)
RMSE_test = np.sqrt(mean_squared_error(Y_test, test_pred))

print('\n')
print('Learning results for test set (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_test)
print('RMSE score: ', RMSE_test)

### plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_test, test_pred, 70, 'tab:red', alpha=0.2)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel(r'Measured $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
plt.ylabel(r'Predicted $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
ticker_arg = [10, 20, 10, 20]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(16)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(16)) for j in ycoord]
dpi_assign = 500
plt.savefig('rf_fus_fig3.jpg', dpi=dpi_assign, bbox_inches='tight')


# ##### FIGURE FOR MANUSCRIPT #####

# ### plot the figure -- TRAIN TEST
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(Y_train, train_pred, 100, 'blue', alpha=0.1)
# ax.scatter(Y_test, test_pred, 100, 'red', alpha=0.2)
# ax.plot([1, 89], [1, 89], "k--", lw=2)
# plt.text(0.03, 0.92, 'Train $RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.85, 'Test $RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.78, 'Train $R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.71, 'Test $R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.xlabel(r'Measured $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'Predicted $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# ticker_arg = [10, 20, 10, 20]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([-3, 93, -3, 93])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# dpi_assign = 500
# plt.savefig('fig_1d.jpg', dpi=dpi_assign, bbox_inches='tight')

# ### plot the figure -- CROSS VALIDATION
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(Y_train, cv_pred, 50, 'purple', alpha=0.2)
# ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
# import matplotlib.font_manager as fm
# fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
# plt.text(0.03, 0.92, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.85, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.xlabel(r'Measured $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# plt.ylabel(r'Predicted $\Delta_{fus}H$ (kJ/mol)', labelpad=10, fontproperties=fonts)
# ticker_arg = [10, 20, 10, 20]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# [(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
# [(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
# dpi_assign = 500
# plt.savefig('fig_S2d.jpg', dpi=dpi_assign, bbox_inches='tight')

# ##### TERMINATE #####


### extract and visualize feature importances
feature_importances = pd.DataFrame([X_train.columns, model.feature_importances_]).T
feature_importances.columns = ['features', 'importance']

print('\n')
print(feature_importances)

fig = plt.figure(figsize=(30,40))
ax = sns.barplot(x=feature_importances['importance'], y=feature_importances['features'], palette='viridis')
fonts = fm.FontProperties(family='arial', size=60, weight='normal', style='normal')
plt.xlabel('Importance', labelpad=10, fontproperties=fonts)
plt.ylabel('Features', labelpad=10, fontproperties=fonts)
import matplotlib.ticker as mticker
ticker_arg = [0.005, 0.03]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
xcoord = ax.xaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(40)) for i in xcoord]
dpi_assign = 500
plt.savefig('rf_fus_fig4.jpg', dpi=dpi_assign, bbox_inches='tight')


# ##### PREDICTION
# """
# Prediction of fusion enthalpy
# <Please make sure the file "candidate_desc.csv" is available before executing this section>
# From here, the script excecutes prediction of fusion enthalpy of 110 pure compounds consisting 60 HBA and 50 HBD
# <Combining the 60 HBA and 50 HBD, we will get 3000 DES combinations>
# Remark: most of the pure compound structures are hypothetical!

# """

# ### load descriptors for prediction
# prediction = pd.read_csv("candidate_desc.csv") # this contains 110 pure compounds
# descriptors = prediction.drop('name', axis=1)

# print('\n')
# print('Descriptor data: \n')
# print(f'Filetype: {type(descriptors)}, Shape: {descriptors.shape}')
# print(descriptors.head())
# print(descriptors.describe())

# ### Predict the melting temperatures of 110 pure compounds
# value_pred = model.predict(descriptors)

# print('\n')
# print('Prediction of descriptor data: ')
# print(value_pred)

# ### export data into a csv file
# np.savetxt('predicted-fus_rf.csv', value_pred, delimiter=',')