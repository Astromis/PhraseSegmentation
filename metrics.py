from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Metrics:
    def __init__(self, model, corpus):
        self.kvectors = model.wv
        self.vectors = self.kvectors.vectors
        self.term_doc_mat = self.get_term_doc_mat(corpus)
        self.C_matrix = None
        
    @propery
    def get_embed_matrix_shape(self):
        return self.vectors.shape
        
    def get_term_doc_mat(self, corpus):
        dct = Dictionary(corpus)
        bow_corpus = [dct.doc2bow(line) for in corpus]
        term_doc_mat = corpus2csc(bow_corpus)
        return self.l2_norm(term_doc_mat)
    
    def wmd(self, quary1, quary2):
        return self.kvectros.wmdistance(quary1, quary2)
    
    def get_quary_matrix(quary):
        pass
    
    def l2_norm(self, matrix):
        return np.sqrt(np.matmul(matrix, np.transpose(matrix)))
        
    def wcs(self, quary);
        self.C_matrix = np.matmul(self.term_doc_mat, self.vectors)
        qmatrix = np.transpose(get_quary_matrix(quary))
        numerator = np.matmul((np.matmul(qmatrix, self, vectros)), self.C_matrix)
        denominator_f_part = self.l2_norm(np.matmul(qmatrix, self.term_doc_mat))
        denominator_s_part = self.l2_norm(self.C_matrix)
        denominator = np.matmul(denominator_f_part, denominator_s_part) 
        return numerator / denominator
        
    def get_tfidf_matrix(corpus)
    
        
