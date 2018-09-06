# -*- coding: utf-8 -*-
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import numpy as np
import json
from read_corpus import read_corpus_from_list

class Metrics:
    def __init__(self, model, corpus):
        self.kvectors = model.wv
        self.vectors = self.kvectors.vectors
        self.term_doc_mat = self.get_term_doc_mat(corpus)
        self.C_matrix = None

    @property
    def get_embed_matrix_shape(self):
        return self.vectors.shape

    def get_term_doc_mat(self, corpus):
        dct = Dictionary(corpus)
        bow_corpus = [dct.doc2bow(line) for line in corpus]
        term_doc_mat = corpus2csc(bow_corpus)
        return self.l2_norm(term_doc_mat)

    def wmd(self, quary1, quary2):
        return self.kvectors.wmdistance(quary1, quary2)

    def get_quary_matrix(quary):
        pass

    def l2_norm(self, matrix):
        #print type(matrix)
        return matrix.dot(matrix.transpose()).sqrt()
        #return np.sqrt(np.matmul(matrix, np.transpose(matrix)))

    def wcs(self, quary):
        self.C_matrix = self.term_doc_mat.dot(self.vectors)
        #print self.vectors.shape
        qmatrix = np.transpose(get_quary_matrix(quary))
        numerator = np.matmul((np.matmul(qmatrix, self, vectros)), self.C_matrix)
        denominator_f_part = self.l2_norm(np.matmul(qmatrix, self.term_doc_mat))
        denominator_s_part = self.l2_norm(self.C_matrix)
        denominator = np.matmul(denominator_f_part, denominator_s_part)
        return numerator / denominator

    def get_tfidf_matrix(corpus):
        pass

model = gensim.models.Word2Vec.load("models/word2vec/size-256_min-count-2_epoch-50_examples-total_window-15_sentences/word2vec_size-100_window-5_min-count-1_workers-4.model")

dataset = json.load(open("./datasets/dataset_paragraphs.json"))
dataset = list(read_corpus_from_list(dataset, tokens_only=True))
dct = Dictionary(dataset)
bow_corpus = [dct.doc2bow(line) for line in dataset]
model =
model = gensim.models.Word2Vec(size=100, window=5, min_count=1, workers=4)
#model.build_vocab(dataset)
model.build_vocab_from_freq(bow_corpus)
model.train(dataset,total_examples=model.corpus_count, epochs=1)
print dir(model.vocabulary)
m = Metrics(model, dataset)
print m.wcs(u"наука")
