# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 09:32:17 2016

@author: ozgur
"""
NUMBER = 5
ThOccurance=10
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import cross_val_score

import numpy as np

import re
import os, sys
import glob
import operator
import pickle
#Tokenizer
#tokenizer = RegexpTokenizer(r'\w+')


def preprocessing(fileN):
	tokenizer = RegexpTokenizer(r'\w+')
	stemmer = PorterStemmer()
	stopWords = stopwords.words("english")
	text = ""

	result = []


	patchNumbers = r'[0-9]+'
	patchEmail = r'^(\s)[a-zA-Z0-9+_\-\.]+[\s]*@[\s]*[0-9a-zA-Z][.-0-9a-zA-Z]*[\s]*.[\s]*[a-zA-Z]+'
	patchDollar = r'[\$]+'
	patchUrl = r'https?[\s]*:[\s]*/[\s]*/[\s]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])[\s]*)+ [\s]*\.[\s]*\w+[\s]*\.[\s]*[a-zA-Z]+'
	patchUrl2 = r'www[\s]*\.[\s]*[a-zA-Z0-9]+'


	lines = fileN.read().splitlines(True)
	subject = lines[0:]
	body = lines[2:]

		#print("subject: " + subject[0])
		#print("body:  " + body[0])

		#FIND REGEX
		#urls = re.findall(patchUrl, body[0])

	text_replaced = re.sub(patchUrl2, 'XXurl', re.sub(patchUrl, 'XXurl', re.sub(patchDollar, 'XXdollar' ,re.sub(patchEmail, 'XXemail', re.sub(patchNumbers, 'XXnumber', body[0])))))

	#remove stop words
	text = ' '.join([word for word in text_replaced.split() if word not in stopWords])

	text_nospecial = re.sub('[^A-Za-z0-9]+', ' ', text)

	#tokenizer
	tokens  = tokenizer.tokenize(text_nospecial.lower())

	#stemming
	for word in tokens:
		result.append(stemmer.stem(word))


	return result





pathCorpusTrain = "/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/train/"
pathCorpusCrossValid = "/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/crossValid/"
projectPath = "/home/ozgur/Desktop/NLP Project/"

#Loading dictionary
with open("/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/dictionaryFS2"+str(ThOccurance)+".pickle", 'rb') as handle:
  dictionary = pickle.load(handle)
print "Dictionary is loaded!"
dicSize = len(dictionary)
print dicSize

trainingLabels = []
#mailName = []
readFirstElement = True
for mailText in glob.glob(os.path.join(pathCorpusTrain, '*.txt')):

    wordVec = np.zeros([dicSize,1])
    indices = []
    #mailName.append(mailText[len(pathCorpus):])

    with open(mailText,'ru') as f:

        #Preprocessing
        tokens  = preprocessing(f)

        for token in tokens:
            if token in dictionary:
                indices.append(dictionary.index(token))

        for index in indices:
            wordVec[index] += 1

        wordVec = wordVec.T

    #Assigning the label of the mail, if spam, label=1 else label=0
    if "spm" in mailText:
        lb = 1
    else:
        lb = 0
    trainingLabels.append(lb)

    if readFirstElement:
        trainingFeatures = wordVec
        readFirstElement = False
    else:
        trainingFeatures = np.append(trainingFeatures,wordVec, axis=0)

trainingLabels = np.array([trainingLabels])
trainingLabels = trainingLabels.T





crossValidLabels = []
#mailName = []
readFirstElement = True
for mailText in glob.glob(os.path.join(pathCorpusCrossValid, '*.txt')):

    wordVec = np.zeros([dicSize,1])
    indices = []
    #mailName.append(mailText[len(pathCorpus):])

    with open(mailText,'ru') as f:

        #Preprocessing
        tokens  = preprocessing(f)

        for token in tokens:
            if token in dictionary:
                indices.append(dictionary.index(token))

        for index in indices:
            wordVec[index] += 1

        wordVec = wordVec.T

    #Assigning the label of the mail, if spam, label=1 else label=0
    if "spm" in mailText:
        lb = 1
    else:
        lb = 0
    crossValidLabels.append(lb)

    if readFirstElement:
        crossValidFeatures = wordVec
        readFirstElement = False
    else:
        crossValidFeatures = np.append(crossValidFeatures,wordVec, axis=0)



"""
numSpam = np.sum(trainingLabels==1)
print "Data is loaded!"
print "Number of data:" + str(len(trainingLabels))
print "Number of spam mail:" + str(numSpam)
print "Number of ham mail:" + str(len(labels)-numSpam)
"""


"""
#Saving the data
with open('features.pickle', 'wb') as handle:
  pickle.dump(features, handle)
with open('labels.pickle', 'wb') as handle:
  pickle.dump(labels, handle)
with open('mailName.pickle', 'wb') as handle:
  pickle.dump(mailName, handle)
"""


"""
#Reading data from pickle files
with open('features.pickle', 'rb') as handle:
  features = pickle.load(handle)
with open('labels.pickle', 'rb') as handle:
  labels = pickle.load(handle)
with open('mailName.pickle', 'rb') as handle:
  mailName = pickle.load(handle)
"""



#Test train set split
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(features,labels,test_size=0.4)


#Classifier
clf = svm.SVC(kernel='linear', C=1)
#clf.fit(X_train,y_train)
#score = cross_val_score(clf, features,labels, cv=2)
#print "SVM is trained!"

#score = clf.score(X_test,y_test)
#print "score: " + str(score)



#Training
clf.fit(trainingFeatures,trainingLabels)

crossPredict = [clf.predict(x) for x in crossValidFeatures]
    
cRes = crossPredict == crossValidLabels
acc = float(np.sum(cRes))/float(len(crossPredict))
TP = 0
FP = 0
FN = 0
for i in range(len(crossPredict)):
    if ((crossPredict[i] == 1) & (crossValidLabels[i] == 0)):
        FP += 1
        #FPmails.append(mailName[k*crossValidSize+i])
    if (crossPredict[i] ==1) & (crossValidLabels[i] == 1):
        TP += 1
    if (crossPredict[i] == 0) & (crossValidLabels[i] == 1):
        FN += 1
        #FNmails.append(mailName[k*crossValidSize+i])

TN = len(crossPredict)-TP-FP-FN
prec = float(TP)/float((TP+FP))
rec = float(TP)/float((TP+FN))
acc = float(TP+TN)/float(len(crossPredict))    


print "Accuracy: " + str(acc)
print "Precision: " + str(prec)
print "Recall: " + str(rec)

"""
print "FP List"
print FPmails
print "FN List"
print FNmails
"""

