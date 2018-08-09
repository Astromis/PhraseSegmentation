from __future__ import division
import json
from word_embedding_train import read_list_corpus
from collections import Counter
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

dataset = json.load(open("./datasets/dataset_paragraphs.json"))
dataset = list(read_list_corpus(dataset, tokens_only=True))
print "Total amount of paragraphs: %d" % len(dataset)

tmp = 0
max_ = 0
min_ = 1000
dct_start = Counter()
dct_end = Counter()
dct_start_2d = Counter()
dct_end_2d = Counter()
for i in dataset:
    tmp += len(i)
    if len(i) > max_ and len(i):
        max_ = len(i)
    if len(i) < min_ and len(i):
        min_ = len(i)
    if len(i) < 2:
        continue
    dct_start[i[0]] += 1
    dct_start_2d[" ".join(i[:2])] += 1
    dct_end[i[-1]] += 1
    dct_end_2d[" ".join([i[-2],i[-1]])] += 1

print "Mean size of paragraphs: %f" % (tmp/len(dataset))
print "Max size: %d, min size: %d" % (max_, min_)

def printcnt(cnt):
    for i in cnt.most_common(30):
        print "%s\t\t %s" % (i[0], i[1])

print printcnt(dct_start)
print "========"
print printcnt(dct_end)
print "========"
print printcnt(dct_start_2d)
print "========"
print printcnt(dct_end_2d)
