# -*- coding: utf-8 -*-
"""
Machine Learning Regression Model -- Sequential MLP {using ANN}

This script executes learning and prediction of melting point using ANN Sequential Multilayer Perceptrons (seq-MLP)
Dataset: Bradley Melting Point Dataset ("melt_temp_data.csv" ---RDkit--> "melt_temp_desc.csv")
         <Please run the file named "descriptor_mp.py" before executing the present script>
Libraries: Pandas, Numpy, Matplotlib, Scikit-learn, TensorFlow, Keras
Learning Algorithm: Sequential Multilayer Perceptrons

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker

### load and define dataframe for the learning dataset
Learning = pd.read_csv("melt_temp_desc.csv") # this contains 3025 data points

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### define X and Y out of the learning data (X: features, Y: values)
X = Learning.drop('mpK', axis=1)
Y = Learning['mpK']

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

##### BUILD AND EVALUATE THE MODEL -- INITIAL
### build neural networks
from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(200, activation="relu", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1))

### compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

### train and evaluate the model
history = model.fit(X_train, Y_train, epochs=100,
                    validation_split=0.2)

### plot the learning curve
progress = pd.DataFrame(history.history)
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
epoch = np.linspace(1, 100, 100)
ax.plot(epoch, progress['loss'],'b',
        epoch, progress['val_loss'], 'r',
        lw=3, alpha=0.5)
ax.legend(['Training Loss', 'Validation Loss'], prop=fonts)
fonts = fm.FontProperties(family='arial', size=16, weight='normal', style='normal')
plt.xlabel(r'Epoch', labelpad=10, fontproperties=fonts)
plt.ylabel(r'Loss', labelpad=10, fontproperties=fonts)
ticker_arg = [10, 20, 1000, 2000]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
plt.axis([0, 101, 0, 13000])
[x.label1.set_fontfamily('arial') for x in xcoord]
[x.label1.set_fontsize(16) for x in xcoord]
[y.label1.set_fontfamily('arial') for y in ycoord]
[y.label1.set_fontsize(16) for y in ycoord]
plt.savefig('fig_S2c.jpg', dpi=300, bbox_inches='tight')

### evaluate the model on the test set
mse_test = model.evaluate(X_test, Y_test)
print('\n')
print('MSE for Test Set:')
print(mse_test)
##### THE INITIAL MODEL STOP HERE

##### FINE-TUNING NEURAL NETWORK HYPERPARAMETERS
##### Wrap Keras Model in Scikit-Learn
### create a function with a keras model given a set of hyperparameters
def build_model(n_hidden=1, n_neurons=50, input_shape=[208]): # 208 is the number of feature in RDkit 2D desc
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam()
    model.compile(loss="mse", optimizer=optimizer)
    return model

### create KerasRegressor based on build_model() --> sklearn
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model) # now keras_reg behaves like sklearn

### train and evaluate the model for a single run
keras_reg.fit(X_train, Y_train, epochs=100,
              validation_split=0.2,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
mse_test = keras_reg.score(X_test, Y_test)

### optimize hyperparameters using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3, 4, 5],
    "n_neurons": np.arange(1, 200),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, 
                                   scoring="r2", n_jobs=n_jobs, random_state=1)
rnd_search_cv.fit(X_train, Y_train, epochs=100,
                  validation_split=0.2,
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

### show optimization results
rnd_search_cv.best_params_
rnd_search_cv.best_score_
model = rnd_search_cv.best_estimator_.model
##### NOW WE GOT THE FINAL MODEL WITH BEST HYPERPARAMETERS

##### VISUALIZATIONS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
ax.scatter(Y_train, train_pred, 50, 'tab:grey', alpha=0.2)
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured Tm (K)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted Tm (K)', labelpad=10, fontproperties=fonts)
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('mlp_mp_fig2.jpg', dpi=dpi_assign, bbox_inches='tight')

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
ax.scatter(Y_test, test_pred, 50, 'tab:red', alpha=0.2)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
plt.text(0.03, 0.92, '$R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.85, '$RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured Tm (K)', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted Tm (K)', labelpad=10, fontproperties=fonts)
ticker_arg = [50, 100, 50, 100]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(14)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(14)) for j in ycoord]
dpi_assign = 500
plt.savefig('mlp_mp_fig3.jpg', dpi=dpi_assign, bbox_inches='tight')


# ##### FIGURE FOR MANUSCRIPT #####

# ### plot the figure -- TRAIN TEST
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for m in ['top', 'bottom', 'left', 'right']:
#     ax.spines[m].set_linewidth(1)
#     ax.spines[m].set_color('black')
# ax.scatter(Y_train, train_pred, 30, 'blue', alpha=0.1)
# ax.scatter(Y_test, test_pred, 30, 'red', alpha=0.2)
# ax.plot([75, 625], [75, 625], "k--", lw=2)
# plt.text(0.03, 0.92, 'Train $RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.85, 'Test $RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.78, 'Train $R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.text(0.03, 0.71, 'Test $R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
# plt.xlabel('Measured Tm (K)', labelpad=10, fontproperties=fonts)
# plt.ylabel('Predicted Tm (K)', labelpad=10, fontproperties=fonts)
# ticker_arg = [50, 100, 50, 100]
# tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.xaxis.set_minor_locator(tickers[0])
# ax.xaxis.set_major_locator(tickers[1])
# ax.yaxis.set_minor_locator(tickers[2])
# ax.yaxis.set_major_locator(tickers[3])
# xcoord = ax.xaxis.get_major_ticks()
# ycoord = ax.yaxis.get_major_ticks()
# plt.axis([50, 650, 50, 650])
# [x.label1.set_fontfamily('arial') for x in xcoord]
# [x.label1.set_fontsize(16) for x in xcoord]
# [y.label1.set_fontfamily('arial') for y in ycoord]
# [y.label1.set_fontsize(16) for y in ycoord]
# dpi_assign = 500
# plt.savefig('fig_1c.jpg', dpi=dpi_assign, bbox_inches='tight')

# ##### TERMINATE #####


# ##### PREDICTION
# """
# Prediction of melting point
# <Please make sure the file "candidate_desc.csv" is available before executing this section>
# From here, the script excecutes prediction of melting temperatures of 110 pure compounds consisting 60 HBA and 50 HBD
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
# np.savetxt('predicted-mp_mlp.csv', value_pred, delimiter=',')