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
import plotly.express as px
import plotly.io as pio
pio.renderers.default="browser"
'''pio.renderers.default="svg"'''

import re
from bs4 import BeautifulSoup
import distance

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
df_new = df.sample(500, random_state=0)

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
# lemmatization and speech tagging
import spacy
nlp = spacy.load("en_core_web_sm")

# token lemmatized speech
def tls(sen):
    doc = nlp(sen)
    speech = [token.lemma_ for token in doc]
    return speech

from words import sym

def preprocess(q):
    #q = str(q).lower().strip()
    
    for i in range(len(sym)):
        words_ = [word.replace(sym[i], "") for word in q]
    q = ''.join(words_)
    
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
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
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
    q = BeautifulSoup(q, features="html.parser")
    q = q.get_text()
    
    # remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    # using tls
    words__ = tls(q)
    q = ' '.join(words__)
    
    
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
# advanced features
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# token features
def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    token_features = [0.0]*8
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    
    # converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    
    # get the non-stopwords in questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # get the stopwords in questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # get the common non-stopwords from question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # get the common stopwords from question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # get the common tokens from question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    # min max
    # cwc min
    # (num of common words) / min(words in q1,q2)
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    # cwc max
    # (num of common words) / max(words in q1,q2)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    # csc min
    # (num of common stop words) / min(stop words in q1,q2)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # csc max
    # (num of common stop words) / max(stop words in q1,q2)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # ctc min
    # (num of common tokens) / min(tokens in q1,q2)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # ctc max
    # (num of common tokens) / max(tokens in q1,q2)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # first word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features
    

# applying token features
token_features = df_new.apply(fetch_token_features, axis = 1)

df_new['cwc_min'] = list(map(lambda x: x[0], token_features))
df_new['cwc_max'] = list(map(lambda x: x[1], token_features))
df_new['csc_min'] = list(map(lambda x: x[2], token_features))
df_new['csc_max'] = list(map(lambda x: x[3], token_features))
df_new['ctc_min'] = list(map(lambda x: x[4], token_features))
df_new['ctc_max'] = list(map(lambda x: x[5], token_features))
df_new['last_word_eq'] = list(map(lambda x: x[6], token_features))
df_new['first_word_eq'] = list(map(lambda x: x[7], token_features))





# length based features
def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # absolute length difference
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # average token length of both questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    
    # longest substring ratio
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    
    return length_features
    
    


# applying length features
length_features = df_new.apply(fetch_length_features, axis = 1)

df_new['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df_new['mean_len'] = list(map(lambda x: x[1], length_features))
df_new['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))




    
    
from fuzzywuzzy import fuzz    
def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzzy ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    
    # fuzzy partial ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    
    # token sort ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    
    # token set ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    
    return fuzzy_features



# applying fuzzy features
fuzzy_features = df_new.apply(fetch_fuzzy_features, axis = 1)

df_new['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
df_new['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
df_new['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
df_new['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


    
    
    

# --------------------------------------------------
# differentiation plots between various advanced features
sns.pairplot(df_new[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']], hue = 'is_duplicate')
sns.pairplot(df_new[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']], hue = 'is_duplicate')
sns.pairplot(df_new[['last_word_eq', 'first_word_eq', 'is_duplicate']], hue = 'is_duplicate')
sns.pairplot(df_new[['mean_len', 'abs_len_diff', 'longest_substr_ratio', 'is_duplicate']], hue = 'is_duplicate')
sns.pairplot(df_new[['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio', 'is_duplicate']], hue = 'is_duplicate')
# --------------------------------------------------




# --------------------------------------------------
# t-distributed stochastic neighbour embedding t-SNE
# dimensionality reduction for 15 features to 3 features
from sklearn.preprocessing import MinMaxScaler
x = MinMaxScaler().fit_transform(df_new[['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 'mean_len', 'abs_len_diff', 'longest_substr_ratio', 'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']])
y = df_new['is_duplicate'].values


from sklearn.manifold import TSNE
tsne3d = TSNE(
    n_components = 3,
    init = 'random', #pca
    random_state = 101,
    method = 'barnes_hut',
    n_iter = 1000,
    verbose = 2,
    angle = 0.5
).fit_transform(x)


# 2d plot for t-SNE
'''df_new['tsne2d_one'] = tsne2d[:,0]
df_new['tsne2d_two'] = tsne2d[:,1]'''

'''sns.scatterplot(
    x = 'tsne2d_one', y = 'tsne2d_two',
    hue = y,
    data = df_new,
    legend = 'full',
    alpha = 0.3
)'''

# 3d plot for t-SNE
tsne3d_one = tsne3d[:,0]
tsne3d_two = tsne3d[:,1]
tsne3d_three = tsne3d[:,2]

px.scatter_3d(
    df_new,
    x = tsne3d_one, y = tsne3d_two, z = tsne3d_three,
    color=('is_duplicate')
)



    
    
    

# --------------------------------------------------
ques_df = df_new[['question1', 'question2']]
final_df = df_new.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])
# --------------------------------------------------


# --------------------------------------------------
# vectorization of words
from words import words

'''from sklearn.feature_extraction.text import CountVectorizer'''
from sklearn.feature_extraction.text import TfidfVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

# implementing word2vec
from gensim.models import Word2Vec
'''from words import words'''

keys=[]
index=[]
values=[]
tokens = [words]

model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, workers=4)
model.build_vocab(tokens)

model.train(tokens, total_examples=len(questions), epochs=10)

for i in range(0, 2466):
    keys.append(model.wv.index_to_key[i])
    index.append(model.wv.key_to_index[keys[i]])
    values.append(model.wv[keys[i]])

dic = {"keys":keys, "index":index, "vectors":values}

'''value = model.wv.index_to_key[2465]
print(value)'''

# vectorizer model
'''cv = CountVectorizer(max_features=3000)'''
tf_idf = TfidfVectorizer()

values = [str(value) for value in values]
q1_arr, q2_arr = np.vsplit(tf_idf.fit_transform(questions).toarray(), 2)

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
x_train, x_test, y_train, y_test = train_test_split(final_df.iloc[:,0:-1].values,
                                                    final_df.iloc[:,-1].values, test_size=0.3, random_state=0)


from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

n = 5
# models  
sgdc = SGDClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n)
svc = SVC(kernel='rbf')
xgb = XGBClassifier()
nb = GaussianNB()
models = [sgdc, dt, rf, knn, svc, xgb, nb]


# hyperparameter grid
param_grid = {
    "sgdc_cls__loss": [0.1, 1, 10, 100],
    "sgdc_cls__penalty": ['linear', 'poly', 'rbf', 'sigmoid'],
    "sgdc_cls__alpha": [2, 3, 4, 5],
    "sgdc_cls__l1_ratio": ['scale', 'auto'],
    "sgdc_cls__fit_intercept": [True, False],
    "sgdc_cls__max_iter": [1000, 2000, 5000],
    "sgdc_cls__tol": [1e-3, 1e-4, 1e-5],
    
    "dt_cls__criterion": ['gini', 'entropy'],
    "dt_cls__splitter": ['best', 'random'],
    "dt_cls__max_depth": [None, 5, 10, 20],
    "dt_cls__min_samples_split": [2, 5, 10],
    "dt_cls__min_samples_leaf": [1, 2, 4],
    
    "rf_cls__n_estimators": [10, 50, 100, 200],
    "rf_cls__criterion": ['gini', 'entropy'],
    "rf_cls__min_samples_split": [2, 5, 10],
    "rf_cls__min_samples_leaf": [1, 2, 4],
    "rf_cls__min_weight_fraction_leaf": [0, 0.1, 0.2],
    "rf_cls__max_leaf_nodes": [None, 10, 20, 30],
    "rf_cls__max_depth": [None, 5, 10],
    
    "knn_cls__n_neighbours": [3, 5, 7, 9],
    "knn_cls__weights": ['uniform', 'distance'],
    "knn_cls__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    
    "sv_cls__C": [0.1, 1, 10, 100],
    "sv_cls__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "sv_cls__degree": [2, 3, 4, 5],
    "sv_cls__gamma": ['scale', 'auto'],
    
    "xgb_cls__max_depth": [3, 5, 7, 9],
    "xgb_cls__learning_rate": [0.1, 0.2, 0.3],
    "xgb_cls__n_estimators": [100, 200, 300],
    "xgb_cls__gamma": [0, 0.5, 1],
    "xgb_cls__subsample": [0.5, 0.8, 1.0],
    "xgb_cls__colsample_bytree": [0.5, 0.8, 1.0],
    "xgb_cls__reg_alpha": [0, 0.5, 1],
    
    "nb_cls__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    
    "lr_cls__penalty": ['l1', 'l2'],
    "lr_cls__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
param_grid={}


# cross-validating training dataset
'''scores = cross_val_score(pipeline, x_train, y_train, cv=5)'''
kfold = KFold(n_splits=5, shuffle=True, random_state=0)


def combined(scorer_1, scorer_2):
    return 0.5*scorer_1 + 0.5*scorer_2





grid.fit(x_train, y_train)
print(grid.score(x_test, y_test))

# validation on testing data
def cross_val_test():
    scores = []
    for model in models:
        for train, test in kfold.split(x_train):
            
            grid = GridSearchCV(
                model,
                param_grid,
                cv=kfold,
                scoring=['accuracy'], refit='accuracy',
            )
            
            grid.fit(x_train[train], y_train[train])
            score = grid.score(x_train[test], y_train[test])
            scores.append(score)
        print("cv_score : {:.2f} +/- {:.2f}".format(np.mean(scores), np.std(scores)))
        print(grid.best_params_)
    print(grid.best_estimator_)
    print("")

print("----TEST----")
cross_val_test()
# --------------------------------------------------


# selected model
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
log = metrics.log_loss(y_test, y_pred)
mtrx = metrics.confusion_matrix(y_test, y_pred)
print("accuracy : {:.2f} +/- {:.2f}".format(np.mean(acc), np.std(acc)))
print("logloss : {:.2f}".format(np.mean(log)))
print("confusion matrix")
print(mtrx)
print("")
