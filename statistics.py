# -*- coding: utf-8 -*-
import nltk
import string
from nltk.corpus import stopwords
from sentence_extracor import segment_sentences, segment_sentences_tok

def tokenize_me(file_text, process_sents=False, write_sents=False, save_tokens=False, load_tokens=False, verbose=False, depth="raw"):
    tokens = []
    #firstly let's apply nltk tokenization
    if load_tokens:
        print("Loading tokens...")
        f = open("./datasets/tokens.txt")
        for line in f:
            tokens.append(line.decode("utf8")[:-1])
        f.close()
        print("Done!")
    else:
        print("Tokinization...")
        tokens = nltk.word_tokenize(file_text)
    if save_tokens and load_tokens == False:
        print("Saving tokens...")
        f = open("./datasets/tokens.txt", 'w')
        for i in tokens:
            if i == None:
                continue
            f.write(i.encode("utf8") + "\n")
        f.close()
    if len(tokens) == 0:
        print "No tokens is load/tokenized."
        return 0
    if verbose: print("Raw tokens count: %d" % len(tokens))
    if process_sents:
        sents = segment_sentences(tokens)
        if write_sents:
            f = open("sents_file", 'w')
            for sent in sents:
                f.write(sent.encode("utf8") + "\n")
            f.close()
        if verbose: print("Sentence count: %d" % len(sents))
        '''
        #for this code sentences in sents must be a list of tokens wich provided segment_sentences_tok function
        count = 0
        for i in sents:
            count += len(i)
        print("Avarage length of sentence: %d" % (count/len(sents)))
        '''
    if depth == "raw": return tokens

    #let's delete punctuation symbols
    tokens = [i for i in tokens if ( i not in string.punctuation )]
    if verbose: print("Tokens count after removing string punctuatuins: %d" % len(tokens))
    if depth == "punc": return tokens
    #deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend([u'что', u'это', u'так', u'вот', u'быть', u'как', u'в', u'—', u'к', u'на'])
    tokens = [i for i in tokens if ( i not in stop_words )]
    if verbose:print("Tokens count after removing stop-words: %d" % len(tokens))
    if depth == "stopw": return tokens
    #couting unique words
    if verbose: print("Unique tokens count: %d" % len(list(set(tokens))))
    if depth == "uniq": return tokens
    
f = open("./datasets/dataset_without_tags.txt")
data = f.read().decode("utf8")

tokenize_me(data, load_tokens=True)
