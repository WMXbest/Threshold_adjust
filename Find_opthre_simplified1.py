# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:58:13 2024
调用自定义函数，寻找最优阈值
@author: wmxbe
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns

import statsmodels.api as sm
import custom_function as cf

# =============================================================================
# 1.Input training data from source domain.
# =============================================================================
#1.input training data. #问题：需要把训练集全部放进去训练吗？领域间分布差异导致的“数据偏移”（data shift）问题.
np.random.seed(29)
# Generate a dataset with 10000 samples, 2 features, and no imbalance (balancing data is very important in this scene and we have to modefitr this imbalanced data to balanced datain to imbanlanced data
#is very important. in this scene.)
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, class_sep=0.5, weights=[0.5])

#2.pre_trained model
pre_trained_model, X_cali, y_cali, optimal_th, optimal_point,y_pred_prob_train, X_train, y_train=cf.pre_trained_model(X, y, test_size=0.2, random_state=42)      

# =============================================================================
# 3.Use the data from the target domain to fine-tune the model to further improve the performance of the model in the target domain.
# =============================================================================
#3.calibration data fine-tune.
#simulated 100 class '0' data as target domain.
neg=X_train[y_train == 0]

import smote
smote = smote.Smote(N=100)
neg_sim = smote.fit(neg)
neg_sim1=neg_sim[0:100,:]   #select the first 100. 
neg_sim2=neg_sim[0:100,:]+1 #overall plus one.
a=neg_sim2.shape[0] #行 
y_neg_sim2=np.zeros((a,1))  #label of simulated data.

#代入模型
Spe_sim2, y_pred_prob_neg_sim2=cf.test_model(neg_sim2, y_neg_sim2,optimal_th,optimal_point)

#调整阈值
delta=0.001
diff=0.01
thre_final, error_final, Spe_final=cf.Fine_tune(delta,
                                             diff,optimal_point,optimal_th,Spe_sim2,y_pred_prob_neg_sim2,y_neg_sim2)
print(f"thre_final: {thre_final}")
print(f"error_final: {error_final}") 
print(f"Spe_final: {Spe_final}") 


# =============================================================================
# Unseen test data 
# =============================================================================
# Using threshold after fine tuning to substitute into the model
