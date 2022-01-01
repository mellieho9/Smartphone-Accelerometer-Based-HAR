##Import modules

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats
import math
from siml.sk_utils import *
from siml.signal_analysis_utils import *
import sensormotion as sm
from scipy.fftpack import fft
from scipy.signal import welch

from IPython.display import display
import matplotlib.pyplot as plt
%matplotlib inline
import pywt
import time
import datetime as dt
from collections import defaultdict, Counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split

import warnings
warnings.filterwarnings("ignore")

##Load input data

train_ts = pd.read_csv("train_time_series.csv", index_col=0)
train_labels = pd.read_csv("train_labels.csv", index_col=0)

##Creates 4th component from the three orthogonal axes
def magnitude(activity):
    x2 = activity['x'] * activity['x']
    y2 = activity['y'] * activity['y']
    z2 = activity['z'] * activity['z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

pd.DataFrame.from_dict(Counter(train_labels['label']), orient="index", columns=['label']).reset_index()

##Data visualization by timestamp

plt.figure(figsize=(10,3))
plt.plot(train_ts['timestamp'], train_ts['x'], linewidth=0.5, color='r', label='x-component')
plt.plot(train_ts['timestamp'], train_ts['y'], linewidth=0.5, color='b', label='y-component')
plt.plot(train_ts['timestamp'], train_ts['z'], linewidth=0.5, color='g', label='z-component')
plt.xlabel('timestamp')
plt.ylabel('acceleration')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))
ax[0].set_title('X-axis: Side to side motion')
ax[0].plot(train_ts['timestamp'], train_ts['x'], linewidth=0.5, color='r')
ax[1].set_title('Y-axis: Up down motion')
ax[1].plot(train_ts['timestamp'], train_ts['y'], linewidth=0.5, color='b')
ax[2].set_title('Z-axis: Forward backward backward')
ax[2].plot(train_ts['timestamp'], train_ts['z'], linewidth=0.5, color='g')
ax[3].set_title('Magnitude, m: Combined X-Y-Z')
ax[3].plot(train_ts['timestamp'], train_ts['m'], linewidth=0.5, color='k')
fig.subplots_adjust(hspace=.5)

##Data visualization by activity

##Separating data for each activity
train_df = pd.concat([train_ts, train_labels['label']], axis=1).dropna()
columns = ['timestamp', 'x', 'y', 'z', 'm', 'label']

standing = train_df[columns][train_df.label == 1]
walking = train_df[columns][train_df.label == 2]
stairsdown = train_df[columns][train_df.label == 3]
stairsup = train_df[columns][train_df.label == 4]

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activities, titles):
    fig, axs = plt.subplots(nrows=len(activities), figsize=(10, 8))
    for i in range(0, len(activities)):
        plot_axis(axs[i], activities[i]['timestamp'], activities[i]['m'], titles[i])
    plt.subplots_adjust(hspace=0.2)
    plt.show()

plot_activity([standing, walking, stairsdown, stairsup],
              ['Standing', 'Walking', 'Stairs down', 'Stairs sup'])

##Creating training and test sets

train_x_list = [train_ts.x.iloc[start:start+10] for start in range(len(train_labels))]
train_y_list = [train_ts.y.iloc[start:start+10] for start in range(len(train_labels))]
train_z_list = [train_ts.z.iloc[start:start+10] for start in range(len(train_labels))]
train_m_list = [train_ts.m.iloc[start:start+10] for start in range(len(train_labels))]
train_signals = np.transpose(np.array([train_x_list, train_y_list, train_z_list, train_m_list]), (1, 2, 0))
train_labels = np.array(train_labels['label'].astype(int))

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
no_labels = len(np.unique(train_labels[:]))

##Randomize training set
def randomize(dataset, labels):
   permutation = np.random.permutation(labels.shape[0])
   shuffled_dataset = dataset[permutation, :]
   shuffled_labels = labels[permutation]
   return shuffled_dataset, shuffled_labels

train_signals, train_labels = randomize(train_signals, np.array(train_labels))

##Creating frequency transformation functions
def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [(1/f_s) * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
    
def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features_ft(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y
 
def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            
            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100-percentile)
            #ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min)/denominator
            
            features += get_features_ft(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features_ft(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features_ft(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

##Implementing and visualizing frequency transformation functions
activities_description = {
    1: 'Standing',
    2: 'Walking',
    3: 'Stairs down',
    4: 'Stairs up'
}

N = 10
f_s = 1 #1 Hz for train_labels #10 Hz for train_ts 
t_n = 1 #1 sec for train_labels #0.1 sec for train_ts
T = t_n / N #
sample_rate = 1 / f_s
denominator = 10

labels = ['x-component', 'y-component', 'z-component']
colors = ['r', 'g', 'b']
suptitle = "Different signals for the activity: {}"
 
xlabels = ['Time [sec]', 'Freq [Hz]', 'Freq [Hz]', 'Time lag [s]']
ylabel = 'Amplitude'
axtitles = [['Standing: Acc', 'Walking: Acc', 'Stairs dn: Acc', 'Stairs up: Acc'],
            ['Standing: FFT Acc', 'Walking: FFT Acc', 'Stairs dn: FFT Acc', 'Stairs up: FFT Acc'],
            ['Standing: PSD Acc', 'Walking: PSD Acc', 'Stairs dn: PSD Acc', 'Stairs up: PSD Acc'],
            ['Standing: Autocorr Acc', 'Walking: Autocorr Acc', 'Stairs dn: Autocorr Acc', 'Stairs up: Autocorr Acc']
           ]

list_functions = [get_values, get_fft_values, get_psd_values, get_autocorr_values]
signal_no_list = [5, 20, 160, 120]
activity_name = list(activities_description.values())

f, axarr = plt.subplots(nrows=4, ncols=4, figsize=(12,8))
f.suptitle(suptitle.format(activity_name), fontsize=10)
 
for row_no in range(0,4):
    for col_no in range(0,4):
        for comp_no in range(0,3):
            color = colors[comp_no % 3]
            label = labels[comp_no % 3]

            axtitle  = axtitles[row_no][col_no]
            xlabel = xlabels[row_no]
            value_retriever = list_functions[row_no]

            ax = axarr[row_no][col_no]
            ax.set_title(axtitle, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=10)
            
            if col_no == 0:
                ax.set_ylabel(ylabel, fontsize=10)

            signal_no = signal_no_list[col_no]
            signals = train_signals[signal_no, :, :]
            signal_component = signals[:, comp_no]
            x_values, y_values = value_retriever(signal_component, T, N, f_s)
            ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
            
            if row_no > 0:
                max_peak_height = 0.1 * np.nanmax(y_values)
                indices_peaks = detect_peaks(y_values, mph=max_peak_height)
                ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
            if col_no == 3:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.6)
plt.show()

##Extracting transformed features
X_train_ft, Y_train_ft = extract_features_labels(train_signals, train_labels, T, N, f_s, denominator)

##Training classifiers
X_train, X_val, Y_train, Y_val = train_test_split(X_train_ft, Y_train_ft, train_size=0.8, random_state=1)
models = batch_classify(X_train, Y_train, X_val, Y_val)
display_dict_models(models)


##Randomizing to address class imbalance
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, Y_ros = ros.fit_sample(X_train, Y_train)
models = batch_classify(X_ros, Y_ros, X_val, Y_val)
display_dict_models(models)

##Parameter tuning
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=7, 
                               n_jobs = -1)

# Fit the random search model
rf_random.fit(X_ros, Y_ros)

## Displaying model performance

clf = rf_random.best_estimator_
clf.fit(X_ros, Y_ros)
print("Accuracy on training set is : {}".format(clf.score(X_ros, Y_ros)))
print("Accuracy on validation set is : {}".format(clf.score(X_val, Y_val)))
Y_val_pred = clf.predict(X_val)
print(classification_report(Y_val, Y_val_pred))


