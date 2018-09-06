import smart_open
import gensim

def read_corpus_from_file(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f: #encoding="iso-8859-1"
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def read_corpus_from_list(list_corp, tokens_only=False):
    for i, paragraph in enumerate(list_corp):
        if tokens_only:
            yield gensim.utils.simple_preprocess(paragraph[0])
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(paragraph[0]), [i])
