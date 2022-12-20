# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:59:01 2022

@author: VAGUE
"""

import numpy as np
'''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)'''
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=1000)

from main import preprocess
from main import common_words
from main import total_words

from main import fetch_token_features
from main import fetch_length_features
from main import fetch_fuzzy_features

def query_point_creator(q1, q2):
    input_query = []
    
    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(common_words(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2)/total_words(q1, q2), 2))
    
    # fetch token features
    token_features = fetch_token_features(q1, q2)
    input_query.extend(token_features)
    
    # fetch length based features
    length_features = fetch_length_features(q1, q2)
    input_query.extend(length_features)
    
    # fetch fuzzy features
    fuzzy_features = fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)
    

    # bow feature for q1
    q1_bow = tf_idf.transform([q1]).toarray()
    
    # bow feature for q2
    q2_bow = tf_idf.transform([q2]).toarray()
    
    return np.hstack((np.array(input_query).reshape(1,22), q1_bow, q2_bow))





q1 = 'Is this the shit?'
q2 = 'Is this Shit?'

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

rf_cls = RandomForestClassifier()
pred1 = rf_cls.predict(query_point_creator(q1, q2))
print(pred1)

svc_cls = SVC()
pred2 = svc_cls.predict(query_point_creator(q1, q2))
print(pred2)