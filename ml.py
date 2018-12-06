# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:49:41 2018

@author: Ed432
"""
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('final_data.csv')

x = df.drop('label', axis = 1)
x = x.drop('patient',axis = 1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
