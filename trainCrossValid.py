# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:12:24 2016

@author: ozgur
"""

from sklearn.utils import shuffle
import os, sys
import glob
from shutil import copyfile
import numpy as np


pathCorpus = "/home/ozgur/Desktop/NLP Project/corpusAll/"
projectPath = "/home/ozgur/Desktop/NLP Project/"

mailName= []
onlyName = []
for mailText in glob.glob(os.path.join(pathCorpus, '*.txt')):
    mailName.append(mailText)
    
    
shuffled = shuffle(mailName)
for x in shuffled:
    onlyName.append(x[len(pathCorpus):])
    
    
numFiles = len(shuffled)
crossValidSize = np.round(numFiles/5)
trainSize = numFiles-crossValidSize

fold1 = "/home/ozgur/Desktop/NLP Project/5foldCV/1/crossValid/"
fold2 = "/home/ozgur/Desktop/NLP Project/5foldCV/2/crossValid/"
fold3 = "/home/ozgur/Desktop/NLP Project/5foldCV/3/crossValid/"
fold4 = "/home/ozgur/Desktop/NLP Project/5foldCV/4/crossValid/"
fold5 = "/home/ozgur/Desktop/NLP Project/5foldCV/5/crossValid/"

print numFiles
print crossValidSize
print trainSize

train1 = onlyName[:]
train2 = onlyName[:]
train3 = onlyName[:]
train4 = onlyName[:]
train5 = onlyName[:]

for i in range(crossValidSize):
    dst1 = fold1+onlyName[i]
    dst2 = fold2+onlyName[i+crossValidSize]
    dst3 = fold3+onlyName[i+2*crossValidSize]
    dst4 = fold4+onlyName[i+3*crossValidSize]
    dst5 = fold5+onlyName[i+4*crossValidSize]
    copyfile(shuffled[i], dst1)
    copyfile(shuffled[i+crossValidSize], dst2)
    copyfile(shuffled[i+2*crossValidSize], dst3)
    copyfile(shuffled[i+3*crossValidSize], dst4)
    copyfile(shuffled[i+4*crossValidSize], dst5)
    del train1[0]
    del train2[crossValidSize]
    del train3[2*crossValidSize]
    del train4[3*crossValidSize]
    del train5[4*crossValidSize]

fold1 = "/home/ozgur/Desktop/NLP Project/5foldCV/1/train/"
fold2 = "/home/ozgur/Desktop/NLP Project/5foldCV/2/train/"
fold3 = "/home/ozgur/Desktop/NLP Project/5foldCV/3/train/"
fold4 = "/home/ozgur/Desktop/NLP Project/5foldCV/4/train/"
fold5 = "/home/ozgur/Desktop/NLP Project/5foldCV/5/train/"

for i in range(trainSize):
    dst1 = fold1+train1[i]
    dst2 = fold2+train2[i]
    dst3 = fold3+train3[i]
    dst4 = fold4+train4[i]
    dst5 = fold5+train5[i]
    copyfile(pathCorpus+train1[i], dst1)
    copyfile(pathCorpus+train2[i], dst2)
    copyfile(pathCorpus+train3[i], dst3)
    copyfile(pathCorpus+train4[i], dst4)
    copyfile(pathCorpus+train5[i], dst5)
 