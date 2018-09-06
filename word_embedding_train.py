# -*- coding: utf-8 -*-
from __future__ import division
import gensim
import nltk
import smart_open
import json
from sentence_extracor import segment_sentences_tok
from read_corpus import read_list_corpus
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

'''
#this block is for buildeing sentence embeddings
#f = open("dataset.txt")
#data = f.read().decode("utf8")
#f.close()

#tokens = nltk.word_tokenize(data)
#sents = segment_sentences_tok(tokens)
'''
dataset = json.load(open("./datasets/dataset_paragraphs.json"))

'''
model = gensim.models.Word2Vec(size=256, window=15, min_count=2, workers=4) #window=10 window=20
model.build_vocab(sents)
print("String training word2vec model...")
model.train(sents, total_examples=model.corpus_count, epochs=50)
model.save("./word2vec_size-100_window-5_min-count-1_workers-4.model")

model = gensim.models.FastText(size=256, window=15, min_count=2, workers=4) #window=10 window=20
model.build_vocab(sents)
print("String training fasttext model...")
model.train(sents, total_examples=model.corpus_count, epochs= 50)
model.save("./fasttext")


#train_corpus = list(read_corpus('sents_file.txt'))
train_corpus = list(read_list_corpus(dataset))
model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=2, workers=4)
model.build_vocab(train_corpus)
print("Starting training doc2vec model...")
model.train(train_corpus, total_examples=model.corpus_count, epochs=50)
model.save('./my_model.doc2vec')
'''
#dataset = list(read_corpus('sents_file.txt', tokens_only=True))
dataset = list(read_list_corpus(dataset, tokens_only=True))
dct = Dictionary(dataset)
corpus = [dct.doc2bow(line) for line in dataset]
model = TfidfModel(corpus)
matrix = model[corpus]
print dir(matrix)
#model.save("./tfidf")
