# -*- coding: utf-8 -*-
__author__ = 'Arturo'

import csv
import numpy as np
import sklearn
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix


#global
regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
stop_words = set(stopwords.words('spanish')) | set(stopwords.words('english'))
porter = PorterStemmer()


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def preprocessing(job_post):
    sentence = job_post

    #remove punctuation and split into seperate words
    global regex_punctuation
    sentence = regex_punctuation.sub(' ', sentence)

    #Remove numbers and other characters
    sentence = re.sub('[0-9]', '', sentence)
    sentence = re.sub(ur'(?iu)[¿¡´`“”►«»·]', '', sentence.decode('utf-8')).encode('utf-8')

    #Para convertir a minusculas las palabras acentuadas y la Ñ + limpieza final
    sentence = sentence.decode('utf-8').lower().encode('utf-8')
    sentence = re.sub(ur'(?iu)[^a-z0-9áéíóúñü ]', '', sentence.decode('utf-8')).encode('utf-8')

    #filter stopwords and stemming
    global porter
    global stop_words
    words = [porter.stem(w.decode('utf-8')).encode('utf-8') for w in sentence.split() if w not in stop_words]

    sentence = " ".join(words)
    return words, sentence


def tfidf_vect(corpus):
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    y = vectorizer.get_feature_names()
    return X, y


def main(csvfile = "TA_Registros_etiquetados.csv"):
    corpus = []
    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            w, s = preprocessing(" ".join(row[4:]))
            corpus.append(s)

    X, y = tfidf_vect(corpus)
    print len(y), y[1000], X[0,1000]
    print type(X), X.shape
    save_sparse_csr("corpus_tfidf",X)
    X = load_sparse_csr("corpus_tfidf.npz")
    print type(X), X.shape, X[0,1000]


main()
