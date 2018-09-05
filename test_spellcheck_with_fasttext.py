# -*- coding: utf-8 -*-
from __future__ import division
import gensim
import nltk
import smart_open
import json
import numpy as np
from sentence_extracor import segment_sentences_tok
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
    
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f: #encoding="iso-8859-1"
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
                
def read_list_corpus(list_corp, tokens_only=False):
    for i, paragraph in enumerate(list_corp):
        if tokens_only:
            yield gensim.utils.simple_preprocess(paragraph[0])
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(paragraph[0]), [i])
            
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
    
def get_graph(X, X1, data):
    X_embedded = TSNE(n_components=2).fit_transform(X[:1000])
    X_embedded1 = TSNE(n_components=2).fit_transform(X1[:1000])
    plt.scatter(X_embedded[:,0],X_embedded[:,1])
    plt.scatter(X_embedded1[:,0],X_embedded1[:,1], color='red')
    for i,word in enumerate(data):
        try:
            plt.annotate(word, X_embedded[i])
            plt.annotate(corrupt(word), X_embedded1[i])
        except:
            continue
    plt.show()
    

def corrupt(word):
    first = word[0]
    sec = word[1]
    #print word + " : " + first + sec + word[3:]
    return first + sec + word[3:]
    
f = open("nitshe.txt")
data = f.read().decode("utf8")
f.close()

tokens = nltk.word_tokenize(data)
sents = segment_sentences_tok(tokens)
data = list(set(data.split(' ')))

model = gensim.models.FastText.load("./fasttext")
vectors = list()
vectors_cor = list()
count = 0
for i in data:
    if len(i) < 5:
        continue
    try:
        vectors.append(model.wv.get_vector(i))
        vectors_cor.append(model.wv.get_vector(corrupt(i)) * similar(i, corrupt(i)))
        if count < 5:
            print corrupt(i)
            print model.most_similar(corrupt(i))[0][0]
            count += 1
    except:
        continue

vectors = np.stack(list(vectors), axis=0)
vectors_cor = np.stack(vectors_cor)

get_graph(vectors, vectors_cor, data)
#model = gensim.models.FastText(size=256, window=3, min_count=2, workers=4) #window=10 window=20
#model.build_vocab(sents)
#print("String training fasttext model...")
#model.train(sents, total_examples=model.corpus_count, epochs= 50)
#model.save("./fasttext")


#print(model.wv['Truth'])
#word_vectors = model.wv
#print(word_vectors.shape)
