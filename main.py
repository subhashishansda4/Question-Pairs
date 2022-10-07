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




# bag of words
# --------------------------------------------------
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