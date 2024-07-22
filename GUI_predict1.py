# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:17:36 2024

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
import custom_function1 as cf

import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import classification_report

class LogisticRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fine-tune--Logistic Regression Classifier")

        # 1.文件选择按钮
        self.load_button = tk.Button(root, text="加载源数据", command=self.load_data)
        self.load_button.pack(pady=5)

        # 2.预训练模型按钮
        self.train_button = tk.Button(root, text="预训练模型", command=self.train_model)
        self.train_button.pack(pady=5)
        
        # 3.加载外地数据按钮
        self.load_target_button = tk.Button(root, text="加载目标数据", command=self.load_target_data)
        self.load_target_button.pack(pady=5)
        
        # 4.Fine-tune按钮
        self.finetune_button = tk.Button(root, text="Fine-tune", command=self.fine_tune)
        self.finetune_button.pack(pady=5)
        
        # 5.预测按钮
        self.predict_button = tk.Button(root, text="进行预测", command=self.predict)
        self.predict_button.pack(pady=5)

        # 文本框显示结果
        self.result_text = tk.Text(root, height=20, width=80)
        self.result_text.pack(pady=10)
        
        self.data = None
        self.data1 = None
        self.model = None
        self.scaler = None
        self.optimal_th = None
        self.optimal_point = None

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("加载源数据", "源数据加载成功！")

    def train_model(self):
        if self.data is not None:
            X = self.data.iloc[:, :-1].values
            y = self.data.iloc[:, -1].values

            self.pre_trained_model, X_cali, y_cali, self.optimal_th, self.optimal_point, y_pred_prob_train, X_train, y_train = cf.pre_trained_model(X, y, test_size=0.2, random_state=42)  
            self.model = self.pre_trained_model
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Optimal Threshold: {self.optimal_th}\nOptimal Point: {self.optimal_point}")
            messagebox.showinfo("训练模型", "模型训练完成！")
        else:
            messagebox.showerror("错误", "请先加载数据！")
                  
    def load_target_data(self):
        file_path1 = filedialog.askopenfilename()
        if file_path1:
            self.data1 = pd.read_csv(file_path1)
            messagebox.showinfo("加载目标数据", "目标数据加载成功！")
            
    def fine_tune(self):
        if self.model is not None and self.data1 is not None and self.optimal_th is not None and self.optimal_point is not None:
            X1 = self.data1.iloc[:, :-1].values
            y1 = self.data1.iloc[:, -1].values
            #
            y_pred_prob_neg_sim2 = self.model.predict_proba(X1)[:, 1]
            predict_label_sim2,AUC_sim2,fpr_sim2,tpr_sim2 = cf.label_optimal_th(self.optimal_th, 
                                                                             y_pred_prob_neg_sim2,y1)
            predict_label_sim2 =predict_label_sim2.squeeze()
            aaa=pd.Series(predict_label_sim2).value_counts()
            TN=aaa[0]
            FP=aaa[1]
            Spe_sim2=TN/(TN+FP)
            #or one line code
            # Spe_sim2, y_pred_prob_neg_sim2 = cf.test_model(X1, y1, self.optimal_th)
            
            delta = float(input("请输入阈值累加减值(0.001)："))
            diff = float(input("请输入误差(0.01)："))
            
            self.thre_final, error_final, Spe_final = cf.Fine_tune(delta, diff, self.optimal_point, self.optimal_th, Spe_sim2, y_pred_prob_neg_sim2, y1)
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Final Threshold: {self.thre_final}\nError Final: {error_final}\nSpecificity Final: {Spe_final}")
            messagebox.showinfo("Fine-tune", "Fine-tune 完成！")
        else:
            messagebox.showerror("错误", "请先加载数据并训练模型！")

    def predict(self):
        if self.model is not None:
            predict_window = tk.Toplevel(self.root)
            predict_window.title("输入特征值进行预测")
            
            entries = []
            for i in range(self.data.shape[1] - 1):
                label = tk.Label(predict_window, text=f"特征 {i + 1}")
                label.pack()
                entry = tk.Entry(predict_window)
                entry.pack()
                entries.append(entry)
            
            def make_prediction():
                features = [float(entry.get()) for entry in entries]
                features=np.array(features)
                features = features.reshape(1,-1)
                # features = self.scaler.transform([features])
                prediction_proba = self.model.predict_proba(features)[:,1]
                # 自定义阈值
                threshold = self.thre_final

                # 根据自定义阈值确定最终标签
                y_pred_custom_threshold = (prediction_proba >= threshold).astype(int)
                messagebox.showinfo("预测结果", f"预测类别: {y_pred_custom_threshold}")
                predict_window.destroy()

            predict_button = tk.Button(predict_window, text="预测", command=make_prediction)
            predict_button.pack()

        else:
            messagebox.showerror("错误", "请先训练模型！")

# 创建主窗口并运行应用程序
root = tk.Tk()
app = LogisticRegressionApp(root)
root.mainloop()

