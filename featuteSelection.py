# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 13:54:59 2016

@author: ozgur
"""
#from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import re
import os
import glob
import operator
import pickle

ThOccurance = 10
NUMBER=5
pathCorpus = "/home/ozgur/Desktop/NLP Project/5foldCV/"+str(NUMBER)+"/train/"
projectPath =  "/home/ozgur/Desktop/NLP Project/"


patchNumbers = r'[0-9]+'
patchEmail = r'^(\s)[a-zA-Z0-9+_\-\.]+[\s]*@[\s]*[0-9a-zA-Z][.-0-9a-zA-Z]*[\s]*.[\s]*[a-zA-Z]+'
patchDollar = r'[\$]+'
patchUrl = r'https?[\s]*:[\s]*/[\s]*/[\s]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F])[\s]*)+ [\s]*\.[\s]*\w+[\s]*\.[\s]*[a-zA-Z]+'
patchUrl2 = r'www[\s]*\.[\s]*[a-zA-Z0-9]+'
patchExlamationMark = r'!'
patchSlash = r'/'

#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#stemmer = SnowballStemmer("english",ignore_stopwords=True)
stemmer = PorterStemmer()
stopWords = stopwords.words("english")

numberSpam = 0
numberHam = 0
SPAM = False
ratio = {}
dicSpam = {}
dicHam = {}
dictionary = []
dictionaryFS = []
wordCountSpam = {}
wordCountHam = {}
for mailText in glob.glob(os.path.join(pathCorpus, '*.txt')):
    if "spm" in mailText:
        numberSpam +=1
        SPAM = True
    else:
        SPAM = False
        numberHam+=1
        
    with open(mailText,'ru') as f:
        lines = f.read().splitlines(True)
        
        subject = lines[0:]
        body = lines[2:]
        
        #FIND REGEX
        #urls = re.findall(patchUrl, body[0])

        text_replaced = re.sub(r'_+', ' ', re.sub(patchSlash, 'XXslash', re.sub(patchExlamationMark, 'XXexlamationMark', re.sub(patchUrl2, 'XXurl', re.sub(patchUrl, 'XXurl', re.sub(patchDollar, 'XXdollar' ,re.sub(patchEmail, 'XXemail', re.sub(patchNumbers, 'XXnumber', body[0]))))))))
         #remove stop words
        text = ' '.join([word for word in text_replaced.split() if word not in stopWords])
        
        #tokenizer 
        tokens  = tokenizer.tokenize(text.lower())
 
        if SPAM:
            for word in tokens:
                word = stemmer.stem(word)
                if len(word) >= 2: 
                    if word in dicSpam:
                        dicSpam[word] += 1
                    else:
                        dicSpam[word] = 1
        else:    
            for word in tokens:
                word = stemmer.stem(word)
                if len(word) >= 2: 
                    if word in dicHam:
                        dicHam[word] += 1
                    else:
                        dicHam[word] = 1            




"""
probability calculations 
"""
pS = float(numberSpam)/float(numberSpam+numberHam) 
pH = 1-pS

totalCountsHam = sum(dicHam.itervalues())
totalCountsSpam = sum(dicSpam.itervalues())

for word in dicSpam:
    if dicSpam[word] > ThOccurance:
        dictionary.append(word)
        wordCountSpam[word] = dicSpam[word]

for word in dicHam:
    if dicHam[word] > ThOccurance:
        if word not in dictionary:        
            dictionary.append(word)
        wordCountHam[word] = dicHam[word]

V = len(dictionary)        
""" 
Laplace Smoothing       
"""     
for word in dictionary:
    if word in wordCountSpam:    
        wordCountSpam[word] = float(wordCountSpam[word]+1)/float(totalCountsSpam+V)
    else:
         wordCountSpam[word] = (1.0)/float(totalCountsSpam+V)

    if word in wordCountHam:    
        wordCountHam[word] = float(wordCountHam[word]+1)/float(totalCountsHam+V)
    else:
         wordCountHam[word] = (1.0)/float(totalCountsHam+V)
         
thresTop = 55.0  
thresBottom = 0.03
for word in dictionary:
    ratio[word] = wordCountSpam[word]/wordCountHam[word]
    if (ratio[word]<thresBottom) or (ratio[word]>thresTop):
        dictionaryFS.append(word)


 
sorted_dictSpam = sorted(wordCountSpam.items(), key=operator.itemgetter(1), reverse = True)
sorted_dictHam = sorted(wordCountHam.items(), key=operator.itemgetter(1), reverse = True)
sorted_ratio = sorted(ratio.items(), key=operator.itemgetter(1), reverse = True)

#Saving the data
with open(projectPath+"5foldCV/"+str(NUMBER)+"/dictionaryFS2"+str(ThOccurance)+".txt",'wu') as f:
    f.write('\n'.join('%s' % x for x in dictionary))

with open(projectPath+"5foldCV/"+str(NUMBER)+"/ratio"+str(ThOccurance)+".txt",'wu') as f:
    f.write('\n'.join('%s' % str(x) for x in sorted_ratio))

with open(projectPath+"/5foldCV/"+str(NUMBER)+"/dictionaryFS2"+str(ThOccurance)+".pickle", 'wb') as handle:
  pickle.dump(dictionaryFS, handle)

print "Dictionary Size: " + str(len(dictionaryFS))
