# -*- coding: utf-8 -*-
__author__ = 'Arturo'

import csv
import numpy as np
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

def load_sparse_csr(filename="corpus_tfidf.npz"):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def save_id_labels(filename,arrayId,arrayLabel):
    f = open(filename+".txt","w")
    for idx,id_job in enumerate(arrayId):
        f.write(str(id_job)+","+str(arrayLabel[idx])+"\n")

def load_id_labels(filename="corpus_id_label.txt"):
    return np.loadtxt(filename, dtype=int, delimiter=",")


def save_dict(filename,array):
    #np.savetxt(filename+".txt",array,fmt="%s")
    f = open(filename+".txt","w")
    #f.write("\n".join(array)
    for w in array:
        f.write(w.encode('utf-8')+"\n")
    f.close()

def load_dict(filename="corpus_dict.txt"):
    return np.loadtxt(filename, dtype=str,delimiter="\n")


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
    labels = []
    id_jobs = []
    with open(csvfile, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            id_jobs.append(int(row[0]))
            labels.append(int(row[1]))
            w, s = preprocessing(" ".join(row[3:]))
            corpus.append(s)

    X, y = tfidf_vect(corpus)
    #print len(y), y[1000], X[0,1000]
    #print type(X), X.shape
    #print type(y), np.array(y).shape
    save_sparse_csr("corpus_tfidf",X)
    save_dict("corpus_dict",np.array(y))
    save_id_labels("corpus_id_label",np.array(id_jobs),np.array(labels))
    #X = load_sparse_csr("corpus_tfidf.npz")
    z = load_id_labels()
    print z.shape, z[0]
    #print type(X), X.shape, X[0,1000]
    #print type(y), len(y), y[1000]


main()
