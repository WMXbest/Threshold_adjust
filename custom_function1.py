# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:25:09 2024
python 导入自定义函数
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

#寻找最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

#约登指数寻找最佳阈值
def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds_tr = roc_curve(label, y_prob, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)#计算AUC得分
    optimal_th, optimal_point = Find_Optimal_Cutoff(tpr, fpr, thresholds_tr)
    return fpr, tpr, roc_auc, optimal_th, optimal_point,thresholds_tr

def label_optimal_th(optimal_th, predict_scores, ture_labels,optimal_point):
        
        fpr, tpr, thresholds = roc_curve(ture_labels, predict_scores,drop_intermediate=False)
        AUC = auc(fpr, tpr)
        print("auc2:", auc)
        
        predict_label=[]
        length = len(ture_labels)
        for idx in range(len(ture_labels)):
          predict_score = predict_scores[idx]
          ture_label = ture_labels[idx]
          predict_labels = [1 if score >= optimal_th else 0 for score in predict_scores]
        
        predict_label.append(predict_labels)
        predict_label = np.array(predict_label)
        predict_label=(np.transpose(predict_label))
     
        return predict_label,AUC,fpr,tpr

#predict_label,AUC,fpr,tpr = label_optimal_th(optimal_th, y_predict_prob[:,1],y_test,optimal_point)

def draw_roc_by_sklearn(predict_label, ture_labels):
    #fpr, tpr, thresholds = roc_curve(ture_labels, predict_scores,drop_intermediate=False)
    tpr_o=[]
    fpr_o=[]
    tnr_o=[]
    ppv_o=[]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    acc=[]
    for idx in range(len(ture_labels)):
          predict_labels = predict_label[idx]
          ture_label = ture_labels[idx]         
          if ture_label == 1 and predict_labels == 1:
              tp += 1
          if ture_label == 1 and predict_labels == 0:
              fn += 1
          if ture_label == 0 and predict_labels == 0:
              tn += 1
          if ture_label == 0 and predict_labels == 1:
              fp += 1
    #*******准确率ppv和召回率tpr是最常用的二分类度量*******
    tpr_o = tp / (tp+fn)#sensitivity/ recall(查全率)
    fpr_o = fp / (fp+tn)                       
    tpr_o = np.array(tpr_o)
    fpr_o = np.transpose(fpr_o)
     
    return fpr_o,tpr_o#,tnr_o#, acc, ppv_o,f1
           
def draw_roc_by_sklearn1(predict_label, ture_labels):
    #fpr, tpr, thresholds = roc_curve(ture_labels, predict_scores,drop_intermediate=False)
    tpr_o=[]
    fpr_o=[]
    tnr_o=[]
    ppv_o=[]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    acc=[]
    for idx in range(len(ture_labels)):
          predict_labels = predict_label[idx]
          ture_label = ture_labels[idx]         
          if ture_label == 1 and predict_labels == 1:
              tp += 1
          if ture_label == 1 and predict_labels == 0:
              fn += 1
          if ture_label == 0 and predict_labels == 0:
              tn += 1
          if ture_label == 0 and predict_labels == 1:
              fp += 1
    #*******准确率ppv和召回率tpr是最常用的二分类度量*******
    # tpr_o = tp / (tp+fn)#sensitivity/ recall(查全率)
    fpr_o = fp / (fp+tn)                       
    # tpr_o = np.array(tpr_o)
    fpr_o = np.transpose(fpr_o)
     
    return fpr_o#,tpr_o


def pre_trained_model(X, y, test_size=0.2, random_state=42):
    """Trains a logistic regression model and returns the trained model and cali data."""
    X_train, X_cali, y_train, y_cali = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid, cv=3,scoring="roc_auc")
    grid=grid.fit(X_train, y_train)
    
    # Calculate the AUC score
    y_pred_prob_train = grid.predict_proba(X_train)
       
    joblib.dump(grid,'train_model.m')  #模型保存
    pre_trained_model=joblib.load('train_model.m') #模型从本地调回

    # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds/
    #参数：drop_intermediate=True
    fpr_tr, tpr_tr, roc_auc_tr, optimal_th, optimal_point,thresholds_tr = ROC(y_train, y_pred_prob_train[:,1])
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    
    ecdf0 = sm.distributions.ECDF(y_pred_prob_train[y_train == 0][:,1]) #train
    ecdf1 = sm.distributions.ECDF(y_pred_prob_train[y_train == 1][:,1]) #ecdf0和ecdf1共同作用画出了ROC曲线
    # plt.plot(ecdf0.x, ecdf0.y)
    # plt.title('CDF of training')
    # plt.show()
    
    tpr2=[] #vertical   " Y" coordinate
    fpr2=[] #horizontal " X" coordinate
    # ordered_list = sorted(y_pred_prob_train[:,1]) #阈值从小到大排序
    xd=np.linspace(0, 1, 100)
    for i in xd: 
    # for i in ordered_list:    
        tpr1=1-ecdf1(i)
        fpr1=1-ecdf0(i)
        # tpr1=list(tpr)
        # fpr1=list(fpr)
        tpr2.append(tpr1)
        fpr2.append(fpr1)
            
    plt.plot(fpr2, tpr2, label='ROC-cdf curve',marker='o')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f} \n' 
             f'Sensitivity:{1-ecdf1(optimal_th):.2f} \n'
             f'Specificity: {ecdf0(optimal_th):.2f}',
             verticalalignment='top',
             horizontalalignment='left')
    plt.plot()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    print('Sensitivity/TPR: %.3f' % (1-ecdf1(optimal_th)))
    print('Specificity: %.3f' % ecdf0(optimal_th))
    print('Optimal threshold train: %.3f' % optimal_th) 
    
    return pre_trained_model, X_cali, y_cali, optimal_th, optimal_point,y_pred_prob_train, X_train, y_train

#利用PDF画ROC曲线
def ROC_PDF(y_pred_prob_train,y_train,optimal_th, optimal_point):
    ecdf0 = sm.distributions.ECDF(y_pred_prob_train[y_train == 0][:,1]) #train
    ecdf1 = sm.distributions.ECDF(y_pred_prob_train[y_train == 1][:,1]) #ecdf0和ecdf1共同作用画出了ROC曲线
    
    tpr2=[] #vertical   " Y" coordinate
    fpr2=[] #horizontal " X" coordinate
    # ordered_list = sorted(y_pred_prob_train[:,1]) #阈值从小到大排序
    xd=np.linspace(0, 1, 100)
    for i in xd: 
    # for i in ordered_list:    
        tpr1=1-ecdf1(i)
        fpr1=1-ecdf0(i)
        # tpr1=list(tpr)
        # fpr1=list(fpr)
        tpr2.append(tpr1)
        fpr2.append(fpr1)
            
    plt.plot(fpr2, tpr2, label='ROC-cdf curve',marker='o')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f} \n' 
             f'Sensitivity:{1-ecdf1(optimal_th):.2f} \n'
             f'Specificity: {ecdf0(optimal_th):.2f}',
             verticalalignment='top',
             horizontalalignment='left')
    plt.plot()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    print('Sensitivity/TPR: %.3f' % (1-ecdf1(optimal_th)))
    print('Specificity: %.3f' % ecdf0(optimal_th))
    print('Optimal threshold train: %.3f' % optimal_th) 
  

#寻找最优阈值使得测试集的sen和spe与train的一样
def label_specify_threshhold(optimal_th, predict_scores, ture_labels):
        predict_label1=[]
        length = len(ture_labels)
        for idx in range(len(ture_labels)):
          predict_score = predict_scores[idx]
          ture_label = ture_labels[idx]
          predict_labels = [1 if score >= optimal_th else 0 for score in predict_scores]          
        
        predict_label1.append(predict_labels)
        predict_label1 = np.array(predict_label1)
        predict_label1=(np.transpose(predict_label1))
     
        return predict_label1

#The true labels are all 0. Otherwise, they cannot be used.   
def Fine_tune(delta,diff,optimal_point,optimal_th,Spe_sim2,y_pred_prob_neg_sim2,y_neg_sim2):
    # 寻找最优阈值.
    spe_train=1-optimal_point[0]
    thre_sim_final = None
    error_final = None
    spe_find_final = None
    best_number=None
    if Spe_sim2<spe_train:        
        # thre_sim=optimal_th+delta
        # thre_sim_final=[]
        for i in range(1,1001,1):
            # Calculate the current threshold
            thre_sim=optimal_th+delta*i
            # Predict labels using the current threshold
            predict_label_sim2 = label_specify_threshhold(thre_sim,y_pred_prob_neg_sim2,y_neg_sim2)
            # Calculate the false positive rate using the current prediction
            fpr_o=draw_roc_by_sklearn1(predict_label_sim2, y_neg_sim2)
            
            spe_find=1-fpr_o
            error=abs(spe_train-spe_find)
            if  error<diff:
                thre_sim_final=thre_sim
                error_final=error
                spe_find_final=spe_find
                best_number=i
                break
        # Print the final results
        if thre_sim_final is not None:
            print(f"threshold_cali: {thre_sim_final}")
            print(f"error: {error_final}")
            print(f"specificity: {spe_find}")
        else:
            print("No threshold found that meets the error criteria.")
        
        
    else:
        for i in range(1,1001,1):
            # Calculate the current threshold
            thre_sim=optimal_th-delta*i
            # Predict labels using the current threshold
            predict_label_sim2 = label_specify_threshhold(thre_sim,y_pred_prob_neg_sim2,y_neg_sim2)
            # Calculate the false positive rate using the current prediction
            fpr_o=draw_roc_by_sklearn1(predict_label_sim2, y_neg_sim2)
            
            spe_find=1-fpr_o
            error=abs(spe_train-spe_find)
            if  error<=diff:
                thre_sim_final=thre_sim
                error_final=error
                spe_find_final=spe_find
                best_number=i
                break
        
        # Print the final results
        if thre_sim_final is not None:
            print(f"threshold_cali: {thre_sim_final}")
            print(f"error_final: {error_final}")
            print(f"specificity: {spe_find}") 
            print(f"best_number: {best_number}")
        else:
            print("No threshold found that meets the error criteria.")

               
    return thre_sim_final, error_final, spe_find


# all Label '0' data to find optimal threshold.
def test_model(neg_sim2, y_neg_sim2,threshold_cali,optimal_point):
    """Evaluates the model and prints the accuracy, confusion matrix, and classification report."""
    model=joblib.load('train_model.m') #模型从本地调回
    y_pred_prob_neg_sim2 = model.predict_proba(neg_sim2)[:, 1]
    predict_label_sim2,AUC_sim2,fpr_sim2,tpr_sim2 = label_optimal_th(threshold_cali, 
                                                                     y_pred_prob_neg_sim2,y_neg_sim2,optimal_point)
    predict_label_sim2 =predict_label_sim2.squeeze()
    aaa=pd.Series(predict_label_sim2).value_counts()
    TN=aaa[0]
    FP=aaa[1]
    Spe_sim2=TN/(TN+FP)
    
    return Spe_sim2, y_pred_prob_neg_sim2
