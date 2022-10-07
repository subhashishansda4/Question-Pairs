# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:59:49 2022

@author: VAGUE
"""
# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup

# data import
df = pd.read_csv('train.csv')
df.info()

# -------------------------------------------------
# missing and duplicate values
print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

df.dropna(axis=0, inplace=True)

# check
print('')
print('Missing Values')
print(df.isnull().sum())
print('')
print('Duplicate Rows')
print(df.duplicated().sum())

# distribution of duplicate and non-duplicate questions
print('')
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')

# repeated questions
print('')
print('Repeated Questions')
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions', np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated', x[x].shape[0])
# histogram
print('')
plt.hist(qid.value_counts().values, bins=160)
plt.yscale('log')
plt.show()
# --------------------------------------------------


# --------------------------------------------------
# sample dataset
df_new = df.sample(30000, random_state=0)

# distribution of duplicate and non-duplicate questions
print('')
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')

# repeated questions
print('')
print('Repeated Questions')
qid = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions', np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated', x[x].shape[0])
# histogram
print('')
plt.hist(qid.value_counts().values, bins=160)
plt.yscale('log')
plt.show()
# --------------------------------------------------


# --------------------------------------------------
# data preprocessing
def preprocess(q):
    q = str(q).lower().strip()
    
    # replace certain special characters with their string equivalents
    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # '[math]' appears around 900 times in the whole dataset
    q = q.replace('[math]', '')
    
    # replacing some numbers with string equivalents
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = q.replace(r'([0-9]+)000000000', r'\1b', q)
    q = q.replace(r'([0-9]+)000000', r'\1m', q)
    q = q.replace(r'([0-9]+)000', r'\1k', q)
    
    # decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has / how is / how does",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
    }
    
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
        
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # removing html tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    return q


df_new['question1'] = df_new['question1'].apply(preprocess)
df_new['question2'] = df_new['question2'].apply(preprocess)
# ----------------------------------------------------


# --------------------------------------------------
# feature engineering
# length of string
df_new['q1_len'] = df_new['question1'].str.len()
df_new['q2_len'] = df_new['question2'].str.len()

# number of words
df_new['q1_num_words'] = df_new['question1'].apply(lambda row: len(row.split(" ")))
df_new['q2_num_words'] = df_new['question2'].apply(lambda row: len(row.split(" ")))

# common words
def common_words(row):
    # unique words
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
df_new['word_common'] = df_new.apply(common_words, axis=1)

# total words
def total_words(row):
    # unique words
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))
df_new['word_total'] = df_new.apply(total_words, axis=1)

# words shared
df_new['word_share'] = round(df_new['word_common']/df_new['word_total'], 2)
# --------------------------------------------------


# --------------------------------------------------
# EDA of features
# number of characters
# question 1
sns.displot(df_new['q1_len'])
print('minimum characters', df_new['q1_len'].min())
print('maximum characters', df_new['q1_len'].max())
print('average num of characters', int(df_new['q1_len'].mean()))
# question 2
sns.displot(df_new['q2_len'])
print('minimum characters', df_new['q2_len'].min())
print('maximum characters', df_new['q2_len'].max())
print('average num of characters', int(df_new['q2_len'].mean()))

# number of words
# question 1
sns.displot(df_new['q1_num_words'])
print('minimum words', df_new['q1_num_words'].min())
print('maximum words', df_new['q1_num_words'].max())
print('average num of words', int(df_new['q1_num_words'].mean()))
# question 2
sns.displot(df_new['q2_num_words'])
print('minimum words', df_new['q2_num_words'].min())
print('maximum words', df_new['q2_num_words'].max())
print('average num of words', int(df_new['q2_num_words'].mean()))

# common words
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_common'], label='non duplicate')
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_common'], label='duplicate')
plt.legend()
plt.show()

# total words
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_total'], label='non duplicate')
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_total'], label='duplicate')
plt.legend()
plt.show()

# words shared
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_share'], label='non duplicate')
sns.displot(df_new[df_new['is_duplicate'] == 0]['word_share'], label='duplicate')
plt.legend()
plt.show()
# --------------------------------------------------


# --------------------------------------------------
ques_df = df_new[['question1', 'question2']]
final_df = df_new.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])
# --------------------------------------------------


# --------------------------------------------------
# bag of words
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(), 2)

# converting to dataframe and concatenating
temp_df1 = pd.DataFrame(q1_arr, index = ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index = ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis = 1)
temp_df['is_duplicate'] = df_new['is_duplicate']

# concatenating final_df and ques_df
final_df = pd.concat([final_df, temp_df], axis=1)
# ---------------------------------------------------


# ---------------------------------------------------
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(temp_df.iloc[:,0:-1].values,
                                                    temp_df.iloc[:,-1].values, test_size=0.2, random_state=0)


# accuracy score
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

# DECISION TREE CLASSIFIER
dt_cls = DecisionTreeClassifier()
dt_cls.fit(x_train, y_train)
# predicting using test set
y_pred2 = dt_cls.predict(x_test)
# accuracy score
as2 = metrics.accuracy_score(y_test, y_pred2)
print(as2)
print('')

# RANDOM FOREST CLASSIFIER
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

# scores
models = ['SGDC', 'DTC', 'RFC', 'KNN', 'SVC', 'XGBC']
as_values = [as1, as2, as3, as4, as5, as6]

col = {'Accuracy Values':as_values}
bar = pd.DataFrame(data=col, index=models)
bar.plot(kind='bar')
# --------------------------------------------------


# --------------------------------------------------
# adding more features
# cwc min
# (num of common words) / min(words in q1,q2)


# cwc max
# (num of common words) / max(words in q1,q2)


# csc min
# (num of common stop words) / min(stop words in q1,q2)


# csc max
# (num of common stop words) / max(stop words in q1,q2)


# ctc min
# (num of common tokens) / min(tokens in q1,q2)


# ctc max
# (num of common tokens) / max(tokens in q1,q2)


# last word eq


# first word eq




# mean length


# absolute length difference


# longest substring ratio











