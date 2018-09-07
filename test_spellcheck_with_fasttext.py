# -*- coding: utf-8 -*-

from __future__ import division
import gensim
import numpy as np
import warnings
import statistics
import os 
from sentence_extracor import segment_sentences_tok
from difflib import SequenceMatcher
from matplotlib import pyplot as plt
from matplotlib.cm import rainbow
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#TODO: make a benchmark function, that will examine matching predicted word from corrupted word and original word
def get_fasttest_model(path_to_model, path_to_corpus):
    # Load FastText model from path_to_model.
    # If it doesn't exist, train new model on path_to_data and save it to path_to_model
    
    f = open(path_to_corpus)
    data = f.read().decode("utf8").lower()
    f.close()
    tokens = statistics.tokenize_me(data, depth='raw')
    if os.path.isfile(path_to_model):
        print "Model %s exists. Loading..." % (path_to_model)
        return gensim.models.FastText.load(path_to_model), tokens
    print tokens[:10]
    sents = segment_sentences_tok(tokens)
    print sents[0]
    model = gensim.models.FastText(size=256, window=3, min_count=2, workers=4) #window=10 window=20
    model.build_vocab(sents)
    print("Trainig of FastText model is started...")
    model.train(sents, total_examples=model.corpus_count, epochs= 50)
    model.save(path_to_model)
    return model, tokens
    
def string_similar(a, b):
    #get string similarity. It is not known whether this is the Levenshtein distance.
    
    return SequenceMatcher(None, a, b).ratio()
    

def get_tsne(arrays, data=None, restrict=1000):
    # arrays: list of numpy arrays
    # data: data for annotate. Need for corrupting words only
    # restrict: restrict nubmer of elements of arrays that will be put in tSNE
    
    X_emb = []
    for i in arrays:
        X_emb.append(TSNE(n_components=2).fit_transform(i[:restrict]))
    colors = rainbow(np.linspace(0, 1, len(X_emb)))
    for X, c in zip(X_emb, colors):
        plt.scatter(X[:,0], X[:,1], color=c)
    #this module for annotate corrupted words and original
    if data != None:
        for i,word in enumerate(data):
            try:
                plt.annotate(word, X_emb[0][i])
                plt.annotate(corrupt(word), X_emb[1][i])
            except:
                continue
    plt.show()

def corrupt(word):
    # make sure, that function works correct
    first = word[0]
    sec = word[1]
    #print word + " : " + first + sec + word[3:]
    return first + sec + word[3:]

def main():
    model, tokens = get_fasttest_model("./fasttext", "./nitshe.txt")
    print("Build word vectors...")
    vectors = list()
    vectors_cor = list()
    for i in tokens:
        if len(i) < 5:
            continue
        try:
            vectors.append(model.wv.get_vector(i))
            vectors_cor.append(model.wv.get_vector(corrupt(i)) ) #* string_similar(i, corrupt(i))
        except:
            continue
        #if count < 5:
         #   print corrupt(i)
          #  print model.most_similar(corrupt(i))[0][0]
           # count += 1

    vectors = np.stack(vectors, axis=0)
    vectors_cor = np.stack(vectors)
    print("Print t-SNE...")
    get_tsne([vectors, vectors_cor], tokens) # for right annotating of arrays, a list must have this order
    return 0

if __name__ == "__main__":
    main()

