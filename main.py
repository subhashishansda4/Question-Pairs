# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:59:49 2022

@author: VAGUE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
df.info()

print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

df.dropna(axis=0, inplace=True)

print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

print('')
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')

print('')
print('Repeated Questions')
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions', np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated', x[x].shape[0])

print('')
plt.hist(qid.value_counts().values, bins=160)
plt.yscale('log')
plt.show()




df_new = df.sample(50000)
ques_df = df_new[['question1', 'question2']]

from sklearn.feature_extraction.text import CountVectorizer
questions = list(ques_df['question1']) + list(ques_df['question2'])
cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)

temp_df1 = pd.DataFrame(q1_arr, index = ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index = ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis = 1)
temp_df['is_duplicate'] = df_new['is_duplicate']






from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(temp_df.iloc[:,0:-1].values,
                                                    temp_df.iloc[:,-1].values, test_size=0.2, random_state=0)


#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#rf = RandomForestClassifier()
#rf.fit(x_train, y_train)
#y_pred = rf.predict(x_test)
#print(accuracy_score(y_test, y_pred))

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# STOCHASTIC GRADIENT DESCENT CLASSIFIER
sgdc_cls = SGDClassifier()
sgdc_cls.fit(x_train, y_train)
# predicting using test set
y_pred1 = sgdc_cls.predict(x_test)
# accuracy score
as1 = metrics.accuracy_score(y_test, y_pred1)
print(as1)
print('')

# DECISION TREE REGRESSION
dt_cls = DecisionTreeClassifier()
dt_cls.fit(x_train, y_train)
# predicting using test set
y_pred2 = dt_cls.predict(x_test)
# accuracy score
as2 = metrics.accuracy_score(y_test, y_pred2)
print(as2)
print('')


# RANDOM FOREST REGRESSION
rf_cls = RandomForestClassifier()
rf_cls.fit(x_train, y_train)
# predicting using test set
y_pred3 = rf_cls.predict(x_test)
# accuracy score
as3 = metrics.accuracy_score(y_test, y_pred3)
print(as3)
print('')


# K-NEAREST NEIGHBOUR
n = 5
knn_cls = KNeighborsClassifier(n)
knn_cls.fit(x_train, y_train)
# predicting using test set
y_pred4 = knn_cls.predict(x_test)
# accuracy score
as4 = metrics.accuracy_score(y_test, y_pred4)
print(as4)
print('')


# SUPPORT VECTOR CLASSIFIER
sv_cls = SVC(kernel='rbf')
sv_cls.fit(x_train, y_train)
# predicting using test set
y_pred5 = sv_cls.predict(x_test)
# accuracy score
as5 = metrics.accuracy_score(y_test, y_pred5)
print(as5)
print('')


# XGBOOST CLASSIFIER
xgb_cls = XGBClassifier()
xgb_cls.fit(x_train, y_train)
# predicting using test set
y_pred6 = xgb_cls.predict(x_test)
# accuracy score
as6 = metrics.accuracy_score(y_test, y_pred6)
print(as6)
print('')








models = ['Multiple Linear Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Ridge Regression', 'Bayesian Regression', 'K-Nearest Neighbour', 'Support Vector Regression', 'XGBoost Regression']
as_values = [as1, as2, as3, as4, as5, as6]

col = {'Accuracy Values':as_values}
bar = pd.DataFrame(data=col, index=models)
bar.plot(kind='bar')