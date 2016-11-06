# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:33:30 2016

@author: ozgur
"""
NUMBER = 1
ThOccurance = 10
ThOccuranceBG = 10
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import cross_val_score
from nltk import ngrams
import numpy as np

import re
import os, sys
import glob
import operator
import pickle

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

pathCorpus = "/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/crossValid"
projectPath = "/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/"



#Reading data from pickle files
"""
with open('wordCountHam.pickle', 'rb') as handle:
  wordCountHam = pickle.load(handle)
with open('wordCountSpam.pickle', 'rb') as handle:
  wordCountSpam = pickle.load(handle)
"""
with open(projectPath+'dictionary'+str(ThOccurance)+'.pickle', 'rb') as handle:
  dictionary = pickle.load(handle)
with open(projectPath+'wordCountHamDic'+str(ThOccurance)+'.pickle', 'rb') as handle:
  wordCountHam = pickle.load(handle)
with open(projectPath+'wordCountSpamDic'+str(ThOccurance)+'.pickle', 'rb') as handle:
  wordCountSpam = pickle.load(handle)
with open(projectPath+'pS.pickle', 'rb') as handle:
  pS = pickle.load(handle)
with open(projectPath+'pH.pickle', 'rb') as handle:
  pH = pickle.load(handle)
   
with open("/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/dictionaryBGFS"+str(ThOccuranceBG)+".pickle", 'rb') as handle:
  dictionaryBG = pickle.load(handle)
with open(projectPath+'bgCountHamDic'+str(ThOccurance)+'.pickle', 'rb') as handle:
  ngCountHam = pickle.load(handle)
with open(projectPath+'bgCountSpamDic'+str(ThOccurance)+'.pickle', 'rb') as handle:
  ngCountSpam = pickle.load(handle)
print "Dictionaries are loaded!"
dicSize = len(dictionary)
dicSizeBG = len(dictionaryBG)
print "Token dictionary size: "+str(dicSize)
print "Bigram dictionary size: "+str(dicSizeBG)  
  


labels = []
predict = []
mailName = []
readFirstElement = True
for mailText in glob.glob(os.path.join(pathCorpus, '*.txt')):

    probS = []
    probH = []
    mailName.append(mailText[len(pathCorpus):])

    with open(mailText,'ru') as f:

        #Preprocessing
        tokens  = preprocessing(f)
        bigrams = ngrams(tokens,2)
        for token in tokens:
            if token in dictionary:
                probS.append(wordCountSpam[token])
            if token in dictionary:
                probH.append(wordCountHam[token])
        numTokens=len(probH)
        for ng in bigrams:
            if ng in dictionaryBG:
                probS.append(numTokens+ngCountSpam[ng])
            if ng in dictionaryBG:
                probH.append(numTokens+ngCountHam[ng])
       
    #Assigning the label of the mail, if spam, label=1 else label=0
    if "spm" in mailText:
        lb = 1
    else:
        lb = 0
    labels.append(lb)
    
    logProbS = np.log(probS)
    logProbH = np.log(probH)
    if (sum(logProbS)+np.log(pS))>(sum(logProbH)+np.log(pH)):
       predict.append(1)
    else:
       predict.append(0)

            
accuracy= np.mean(np.array(predict) == np.array(labels))


TP = 0
FP = 0
FN = 0
for i in range(len(predict)):
    if ((predict[i] == 1) & (labels[i] == 0)):
        FP += 1
        #FPmails.append(mailName[k*crossValidSize+i])
    if (predict[i] ==1) & (labels[i] == 1):
        TP += 1
    if (predict[i] == 0) & (labels[i] == 1):
        FN += 1
        #FNmails.append(mailName[k*crossValidSize+i])

TN = len(predict)-TP-FP-FN
prec = float(TP)/float((TP+FP))
rec = float(TP)/float((TP+FN))
#acc = float(TP+TN)/float(len(predict))    


print "Accuracy: " + str(accuracy)
print "Precision: " + str(prec)
print "Recall: " + str(rec)
