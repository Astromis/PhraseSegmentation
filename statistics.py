# -*- coding: utf-8 -*-
import nltk
import string
from nltk.corpus import stopwords
from sentence_extracor import segment_sentences, segment_sentences_tok

def tokenize_me(file_text, process_sents=False, write_sents=False, save_tokens=False, load_tokens=False, verbose=False, depth="raw"):
    """
    Perform tokenization of dataset file with nltk word tokinizer.
        file_text: path to datafile
        process_sents: need to determine sentances
        write_sents: save determined sentences in file
        save_tokens: save tokens in file
        load_tokens: load tokens in file (for later santence segmantation, for example)
        verbose
        depth:  raw - return list of raw tokens withour preprocess
                punc - return list of tokens with delited punctuation        
                stopw - return list of tokens woth delited stop-words
                uniq - return list of unique tokens
    """
    f = open(file_text)
    data = f.read().lower()#.decode("utf8").
    tokens = []
    #firstly let's apply nltk tokenization
    # if dataset was tokenized, try to load tokens
    if load_tokens:
        print("Loading tokens...")
        f = open("./datasets/tokens.txt")
        for line in f:
            tokens.append(line)#.decode("utf8")[:-1] # it no need for python 3.6
        f.close()
        print("Done!")
    else:
        print("Tokinization...")
        tokens = nltk.word_tokenize(data)
    # save tokens in file
    if save_tokens and load_tokens == False:
        print("Saving tokens...")
        f = open("./datasets/tokens.txt", 'w')
        for i in tokens:
            if i == None:
                continue
            f.write(i.encode("utf8") + "\n")
        f.close()
    # if something went wrong
    if len(tokens) == 0:
        print("No tokens is load/tokenized.")
        return 0
    if verbose: print("Raw tokens count: %d" % len(tokens))
    # determine and save sentences in file
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
    # if not need any preprocess on tokens
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
    
tokenize_me("./datasets/dataset_without_tags.txt", load_tokens=True)
