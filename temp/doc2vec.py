import gensim
import os
import collections
import smart_open
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

model = gensim.models.doc2vec.Doc2Vec.load('./my_model.doc2vec')


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f: #encoding="iso-8859-1"
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus('dataset.txt'))
inferred_vector = model.infer_vector(train_corpus[400].words)
print inferred_vector
print train_corpus[10].words


#test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
#lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
#lee_test_file = test_data_dir + os.sep + 'lee.cor'
#train_corpus = list(read_corpus('dataset.txt'))
#test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
#print train_corpus[0]
#next block is training model
'''
model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
model.save('./my_model.doc2vec')
'''

X_embedded = TSNE(n_components=2).fit_transform(vecs)
sec = 3
plt.scatter(X_embedded[:,0],X_embedded[:,1])
plt.scatter(X_embedded[6,0],X_embedded[6,1], c='r')
plt.scatter(X_embedded[sec,0],X_embedded[sec,1], c='y')
plt.show()
