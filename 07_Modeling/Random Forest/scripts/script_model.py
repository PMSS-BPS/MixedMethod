import pandas as pd 
import numpy as np
import math
import itertools
from itertools import compress
import geopandas as gpd

import os
from glob import glob
from tqdm import tqdm
import dtw

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed


from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import time
import math


def plot_confusion_matrix(cm, classes, title, dataset_name):
    plt.figure(figsize=(8, 8))
    
    percentages = (cm.T / cm.sum(axis=1) * 100).T
    
    plt.imshow(percentages, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title(f'Confusion Matrix - {title} - {dataset_name}')
    plt.colorbar(label='Percentage', shrink=0.75)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f"{cm[i, j]}\n{percentages[i, j]:.1f}%", horizontalalignment='center', color='white' if percentages[i, j] > 50 else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

def modeling(Y, X, nama_model, image=False):
    # Stratified split into training and testing
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=1),
    }

    results = {'Model': [], 'Pred Time': [], 'Fit Time': [], 'Mean Accuracy Train (5-fold CV)': [], 
               'Mean Accuracy Test (5-fold CV)': [], 'AUC Train': [], 'AUC Test': [],
               'Accuracy Testing Pred': [], 'Precision Testing Pred': [], 'Recall Testing Pred': []}
    
    accuracy_test_pred_max = 0
    
    for model_name, model in models.items():
        print(f"Processing {model_name}...{nama_model}")

        fit_time_start = time.time()
        model.fit(X_train, Y_train)
        fit_time_end = time.time()

        mean_accuracy_train = cross_val_score(model, X_train, Y_train, cv=5, n_jobs=-1).mean()
        mean_accuracy_test = cross_val_score(model, X_test, Y_test, cv=5, n_jobs=-1).mean()
        
        auc_train = roc_auc_score(Y_train, model.predict_proba(X_train), multi_class='ovr')
        auc_test = roc_auc_score(Y_test, model.predict_proba(X_test), multi_class='ovr')

        pred_time_start = time.time()
        Y_test_pred = model.predict(X_test)
        pred_time_end = time.time()

        cm = confusion_matrix(Y_test, Y_test_pred)
        class_labels = sorted(Y_test.unique())
        plot_confusion_matrix(cm, class_labels, model_name, nama_model)

        accuracy_test_pred = accuracy_score(Y_test, Y_test_pred)
        precision_test_pred = precision_score(Y_test, Y_test_pred, average='weighted')
        recall_test_pred = recall_score(Y_test, Y_test_pred, average='weighted')

        results['Model'].append(f'{model_name} | {nama_model}')
        results['Pred Time'].append(pred_time_end - pred_time_start)
        results['Fit Time'].append(fit_time_end - fit_time_start)
        results['Mean Accuracy Train (5-fold CV)'].append(mean_accuracy_train)
        results['Mean Accuracy Test (5-fold CV)'].append(mean_accuracy_test)
        results['AUC Train'].append(auc_train)
        results['AUC Test'].append(auc_test)
        results['Accuracy Testing Pred'].append(accuracy_test_pred)
        results['Precision Testing Pred'].append(precision_test_pred)
        results['Recall Testing Pred'].append(recall_test_pred)
        
        if accuracy_test_pred > accuracy_test_pred_max:
            accuracy_test_pred_max = accuracy_test_pred
            model_return = model
            df_pred = pd.DataFrame({"y_test": Y_test, "y_pred": Y_test_pred})

    result_df = pd.DataFrame(results)

    ## Kalau model ngga ada feature importancenya, ini hapus aja
    if image:
        plot_feature_importance(model_return, X.columns)
    
    return result_df, model_return, df_pred

