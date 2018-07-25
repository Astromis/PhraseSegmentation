import os

files = os.listdir("./all")
dataset = open("dataset.txt", 'w')
for i in files:
    f = open("./all/"+i)
    for line in f:
        dataset.write(line)
        
