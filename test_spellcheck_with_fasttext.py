# -*- coding: utf-8 -*-

from __future__ import division
import gensim
import numpy as np
import warnings
import statistics
import string
import os 
from sentence_extracor import segment_sentences_tok
from difflib import SequenceMatcher
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist
from matplotlib.cm import rainbow
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# this code must run with python3.x
#TODO: clean corpus from punctuations!
# devide this file
# take a look on paper http://www.dialog-21.ru/media/3450/derezaovetal.pdf and fasttext
# compare self trained fasttext model with facebook provided model
def compute_accuracy(orig, corrected):
    # compute spell correction accuracy
    # input:
    #   orig: list of original words
    #   corrected: list of corrected words
    # return:
    #   correction accuracy by word full match
    res = []
    for i,z in zip(orig, corrected):
        res.append(i == z)
    return sum(res)/len(res)

def baseline_spellchecker(model, tokens, func):
    # This model simply take a most similar word on corrupted word
    original = []
    corrected = []
    for i in tokens:
        try:
            cor_word = func(i)
        except:     
            #print("Exception function!")
            continue
        try:
            corrected_ = model.most_similar(cor_word)[0][0]
        except:
            #print("Exception most_similar!")
            continue
        corrected.append(corrected_)
        original.append(i)
    print(compute_accuracy(original, corrected))

def get_fasttest_model(path_to_model, path_to_corpus):
    # Load FastText model from path_to_model.
    # If it doesn't exist, train new model on path_to_data and save it to path_to_model
    
    #f = open(path_to_corpus)
    #data = f.read().decode("utf8").lower()
    #f.close()
    tokens = statistics.tokenize_me(path_to_corpus, depth='raw')
    if os.path.isfile(path_to_model):
        print("Model %s exists. Loading..." % (path_to_model))
        return gensim.models.FastText.load(path_to_model), tokens
    sents = segment_sentences_tok(tokens)
    model = gensim.models.FastText(size=256, window=3, min_count=2, workers=4) #window=10 window=20
    model.build_vocab(sents)
    print("Trainig of FastText model is started...")
    model.train(sents, total_examples=model.corpus_count, epochs= 50)
    model.save(path_to_model)
    return model, tokens
    
def string_similar(a, b):
    #get string similarity. It is not known whether this is the Levenshtein distance.
    
    return SequenceMatcher(None, a, b).ratio()
    

def matrix_eucl_dist(a,b):
    # compute euclidian distance for rows in matrices
    # input
    #   a, b - numpy matrices
    # return
    #   matrix with shape ("rows of a(b)", 1) 
    c = a - b
    c = np.matmul(c,c.transpose())
    c = np.sum(c, axis=0)
    c = np.sqrt(c)
    return c

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
                plt.annotate(corrupt_random_permutation(word), X_emb[1][i])
            except:
                continue
    plt.show()

def corrupt_random_permutation(word, permut_amount=1):
    corrupted_word = word
    for i in range(permut_amount):
        index = np.random.randint(0, len(corrupted_word)-1)
        first = corrupted_word[index]
        sec = corrupted_word[index+1]
        corrupted_word = corrupted_word[:index] + corrupted_word[index+1] + corrupted_word[index] + corrupted_word[index+2:]
    return  corrupted_word
    
def corrupt_random_insertion(word, insert_amount=1):
    #test this
    corrupted_word = word
    for i in range(insert_amount):
        index = np.random.randint(0, len(corrupted_word)-1)
        first = corrupted_word[index]
        sec = corrupted_word[index]
        insert = np.random.choice(list("qwertyuiopasdfghjklzxcvbnm"))
        corrupted_word = corrupted_word[:index] + insert + corrupted_word[index:]
    #print(corrupted_word)
    return corrupted_word

def corrupt_random_deletion(word, del_amount=1):
    #test this
    corrupted_word = word
    for i in range(del_amount):
        index = np.random.randint(0, len(corrupted_word)-1)
        first = corrupted_word[index]
        sec = corrupted_word[index+1]
        corrupted_word = corrupted_word[:index] + corrupted_word[index+1:]
    return corrupted_word

def main():
    model, tokens = get_fasttest_model("./fasttext", "./nitshe.txt")
    #print("test",corrupt_random_deletion("test"))

    baseline_spellchecker(model, tokens, corrupt_random_insertion)
    
    # code below compute t-SNE for original and corrupted words
    '''
    print("Build word vectors...")
    vectors = list()
    vectors_cor = list()
    entered_words = set()
    for i in tokens:
        if len(i) != 5 and i in entered_words:
            continue
        try:
            vectors.append(model.wv.get_vector(i))
            vectors_cor.append(model.wv.get_vector(corrupt_random_permutation(i)) ) #* string_similar(i, corrupt(i))
            entered_words.add(i)
        except:
            continue
    '''
    #vectors = np.stack(vectors, axis=0)
    #vectors_cor = np.stack(vectors)
    #matrix_eucl_dist(vectors, vectors_cor)
    #plt.show()
    #print("Print t-SNE for %d words..." %(len(list(entered_words))))
    #get_tsne([vectors, vectors_cor], list(entered_words)) # for right annotating of arrays, a list must have this order
    return 0


if __name__ == "__main__":
    main()

