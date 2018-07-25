# -*- coding: utf-8 -*-
import os
import re
import json

CORP_DIR = "./corpora/"
CORP_FILE = "./datasets/dataset.txt"

#list_d = os.listdir(CORP_DIR)
#list_d.remove("parser.py")
'''
for filen in list_d:
    f = open(CORP_DIR+filen)
    for line in f:
        if line.startswith("<pa>"):
            print line[4:]
            break
    break'''
data = open(CORP_FILE).read().decode("utf8")
pattern = re.compile(u"<pa>")
res = re.split(pattern, data)
dataset = []
for i, par in enumerate(res):
    if par[0] == u"<":
        continue
    if par[0] in u"ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ":
        dataset.append([par])
    else: 
        dataset[-1].append(par)
print "Save data..."
json.dump(dataset, open("./datasets/dataset_paragraphs.json", 'w'))
print "Load data..."
another_dataset = json.load(open("./datasets/dataset_paragraphs.json"))
'''
for i in another_dataset:
    print "-----------------------"
    for j in i:
        print j
'''
