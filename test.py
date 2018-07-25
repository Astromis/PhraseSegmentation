import gensim
import json
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc

#model = gensim.models.Word2Vec.load("./models/word2vec/size-256_min-count-2_epoch-50_examples-total_window-15_sentences/word2vec_size-100_window-5_min-count-1_workers-4.model")

def read_list_corpus(list_corp, tokens_only=False):
    for i, paragraph in enumerate(list_corp):
        if tokens_only:
            yield gensim.utils.simple_preprocess(paragraph[0])
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(paragraph[0]), [i])


model = gensim.models.TfidfModel.load("./models/tfidf/sentences/tfidf")
dataset = json.load(open("./datasets/dataset_paragraphs.json"))
dataset = list(read_list_corpus(dataset, tokens_only=True))
dct = Dictionary(dataset)
bow_corpus = [dct.doc2bow(line) for line in dataset]
term_doc_mat = corpus2csc(bow_corpus)
print dir(term_doc_mat)
print term_doc_mat.get_shape()
