# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:59:01 2022

@author: VAGUE
"""


from gensim.models import Word2Vec
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer


import spacy
nlp = spacy.load("en_core_web_sm")

# token lemmatized speech
def tls(sen):
    doc = nlp(sen)
    speech = [token.lemma_ for token in doc]
    return speech

from words import sym
from words import contractions

def preprocess(q):    
    for i in range(len(sym)):
        words_ = [word.replace(sym[i], "") for word in q]
    q = ''.join(words_)
    
    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    q = q.replace('[math]', '')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
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

    q = BeautifulSoup(q, features="html.parser")
    q = q.get_text()

    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    words__ = tls(q)
    q = ' '.join(words__)
    
    return q

def common_words(w1, w2):
    w1 = set(map(lambda word: word.lower().strip(), w1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), w2.split(" ")))
    return len(w1 & w2)

def total_common(w1, w2):
    w1 = set(map(lambda word: word.lower().strip(), w1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), w2.split(" ")))
    return (len(w1) + len(w2)) - (len(w1 & w2))

def total_words(w1, w2):
    w1 = set(map(lambda word: word.lower().strip(), w1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), w2.split(" ")))
    return (len(w1) + len(w2))


import distance
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def fetch_token_features(w1, w2):
    token_features = [0.0]*8
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")

    w1_tokens = w1.split()
    w2_tokens = w2.split()
    
    if len(w1_tokens) == 0 or len(w2_tokens) == 0:
        return token_features
    
    # get the non-stopwords in questions
    w1_words = set([word for word in w1_tokens if word not in STOP_WORDS])
    w2_words = set([word for word in w2_tokens if word not in STOP_WORDS])

    # get the stopwords in questions
    w1_stops = set([word for word in w1_tokens if word in STOP_WORDS])
    w2_stops = set([word for word in w2_tokens if word in STOP_WORDS])
    
    # get the common non-stopwords from question pair
    common_word_count = len(w1_words.intersection(w2_words))
    
    # get the common stopwords from question pair
    common_stop_count = len(w1_stops.intersection(w2_stops))
    
    # get the common tokens from question pair
    common_token_count = len(set(w1_tokens).intersection(set(w2_tokens)))

    token_features[0] = common_word_count / (min(len(w1_words), len(w2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(w1_words), len(w2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(w1_stops), len(w2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(w1_stops), len(w2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(w1_tokens), len(w2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(w1_tokens), len(w2_tokens)) + SAFE_DIV)
    token_features[6] = int(w1_tokens[-1] == w2_tokens[-1])
    token_features[7] = int(w1_tokens[0] == w2_tokens[0])
    
    return token_features


def fetch_length_features(w1, w2):
    length_features = [0.0]*3
    
    w1_tokens = w1.split()
    w2_tokens = w2.split()
    
    if len(w1_tokens) == 0 or len(w2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(w1_tokens) - len(w2_tokens))
    length_features[1] = (len(w1_tokens) + len(w2_tokens)) / 2

    strs = list(distance.lcsubstrings(w1, w2))
    if strs:    
        length_features[2] = len(strs[0]) / (min(len(w1), len(w2)) + 1)
    else:
        length_features[2] = 0
    
    return length_features


from fuzzywuzzy import fuzz
def fetch_fuzzy_features(w1, w2):
    fuzzy_features = [0.0]*4

    fuzzy_features[0] = fuzz.QRatio(w1, w2)
    fuzzy_features[1] = fuzz.partial_ratio(w1, w2)
    fuzzy_features[2] = fuzz.token_sort_ratio(w1, w2)
    fuzzy_features[3] = fuzz.token_set_ratio(w1, w2)
    
    return fuzzy_features




def data_point(q1, q2):
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
    input_query.append(total_common(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2)/total_words(q1, q2), 2))
    
    # fetch token features
    token_features = fetch_token_features(q1, q2)
    input_query.extend(token_features)
    
    # length based features
    length_features = fetch_length_features(q1, q2)
    input_query.extend(length_features)
    
    # fuzzy features
    fuzzy_features = fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)
    
    
    # merge texts
    questions = [q1, q2]
    words = [q.split() for q in questions]

    # implementing word2vec
    w2v = Word2Vec(words, vector_size=100, window=1, min_count=4, workers=4)
    w2v.train(words, total_examples=len(questions), epochs=10)

    # vectorizer model
    tf_idf = TfidfVectorizer(max_features=1000)

    # using word2vec model to transform sentences
    questions_t = []
    for question in questions:
        q_t = []
        for word in question:
            try:
                q_t.append(w2v.wv[word])
            except Exception:
                pass
        questions_t.append(q_t)
        
    sample = []
    for arr in questions_t:
        s = ", ".join(str(x) for x in arr)
        s = s.replace("[", "").replace("]", "")
        sample.append(s)

    # vectors
    q1_arr, q2_arr = np.array_split(tf_idf.fit_transform(sample).toarray(), 2)
    
    return np.hstack((np.array(input_query).reshape(1,23), q1_arr, q2_arr))
    




q1 = "Where is the capital of India?"
q2 = "What is the current capital of India?"


grid = pickle.load(open('grid.pkl', 'rb'))
pred = grid.predict_proba(data_point(q1, q2))[:, 1]
print(pred)

print("Similar") if pred>0.65 else print("Unique")