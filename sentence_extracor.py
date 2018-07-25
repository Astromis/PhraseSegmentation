# -*- coding: utf-8 -*-
import re
import nltk
import pickle
    
def punct_features(tokens, i):
    pattern = re.compile(u'[А-Я]')
    flag = 0
    try:
        return {'next-word-capitalized': re.match(pattern, tokens[i+1]) is not None, #'prev-word': tokens[i-1].lower(), 
                'punct': tokens[i], 'prev-word-is-one-char': len(tokens[i-1]) == 1, 
                'next-word-is-one-char': len(tokens[i+1]) == 1,#'next-word': tokens[i+1]
               }
    except:
        return {'next-word-capitalized': False, #'prev-word': tokens[i-1].lower(), 
                'punct': tokens[i], 'prev-word-is-one-char': len(tokens[i-1]) == 1, 
                'next-word-is-one-char': False,
                #'next-word': tokens[i+1]
               }
        
def train_classifyer():
    f = open('./datasets/sents.txt')
    tokens = []
    boundaries = set()
    offset = 0
    for line in f:
        punct = line[-3].decode("utf8")
        line = line[:-3].decode("utf8")
        line = line.split(' ')
        line.append(punct)
        tokens.extend(line)
        offset += len(line)
        boundaries.add(offset-1)

    featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens)-1) if tokens[i] in u'.?!']

    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print "Training done! Classification accuracy: ", nltk.classify.accuracy(classifier, test_set)
    return classifier

def segment_sentences(words):
    try:
        f = open('sentence_classifyer.pickle', 'rb')
        classifier = pickle.load(f)
        print("Serialized classifyer found. Load...")
        f.close()
    except:
        print("Serialized classifyer not found. Start training classifier...")
        classifier = train_classifyer()
        f = open('sentence_classifyer.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in u'.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(" ".join(words[start:i+1]))
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents
    
def segment_sentences_tok(words):
    try:
        f = open('sentence_classifyer.pickle', 'rb')
        classifier = pickle.load(f)
        print("Serialized classifyer found. Load...")
        f.close()
    except:
        print("Serialized classifyer not found. Start training classifier...")
        classifier = train_classifyer()
        f = open('sentence_classifyer.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in u'.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(list(words[start:i+1]))
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents
