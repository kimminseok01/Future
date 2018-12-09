# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:49:41 2018

@author: Ed432
"""

#import modules
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

#create function for time checking
def start():
    start_time = time.time()
    return start_time

def end(start_time):
    return time.time() - start_time

#import raw data
df = pd.read_csv('final_data_1.csv')

#div data into label and feature
x = df.drop('label', axis = 1)
x = x.drop('patient',axis = 1)
y = df['label']

#div data into test set and train set
#test set is 10% of overall set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

#svm checking
start_time = start()
svmc = svm.SVC()
svmc.fit(x_train, y_train)
pred_1 = svmc.predict(x_test)
end1 = end(start_time)

#random forest checking
start_time = start()
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(x_train, y_train)
pred_2 = rfc.predict(x_test)
end2 = end(start_time)

#gradient boosting checking
start_time = start()
gbc = GradientBoostingClassifier(n_estimators=1000)
gbc.fit(x_train, y_train)
pred_3 = gbc.predict(x_test)
end3 = end(start_time)

#get accuracy and report of each algorithms
accuracy_1 = metrics.accuracy_score(y_test, pred_1)
report_1 = metrics.classification_report(y_test, pred_1)

accuracy_2 = metrics.accuracy_score(y_test, pred_2)
report_2 = metrics.classification_report(y_test, pred_2)

accuracy_3 = metrics.accuracy_score(y_test, pred_3)
report_3 = metrics.classification_report(y_test, pred_3)

#show results
print()
print('Time of supporting vector machine: ', end1)
print('Time of random forest: ', end2)
print('Time of gradient boosting: ', end3)
print()
print('Accuracy of supporting vector machine: ', accuracy_1)
print('Accuracy of random forest: ', accuracy_2)
print('Accuracy of gradient boosting: ', accuracy_3)
print()

print('Supporting Vector Machine'.center(70),'\n',report_1)
print()
print('Random Forest'.center(70),'\n',report_2)
print()
print('Gradient Boosting'.center(70),'\n',report_3)