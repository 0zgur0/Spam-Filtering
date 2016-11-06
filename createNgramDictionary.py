# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:20:47 2016

@author: ozgur
"""

#from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import ngrams
import re
import os
import glob
import operator
import pickle

ThOccurance = 20
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
dicSpam = {}
dicHam = {}
dictionary = []
ngCountSpam = {}
ngCountHam = {}
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
        stemmed = []
        
        if SPAM:
            for word in tokens:
                word = stemmer.stem(word)
                stemmed.append(word)
            
            bigramsSpam = ngrams(stemmed, 2)
            for ng in bigramsSpam:
                if ng in dicSpam:
                    dicSpam[ng]+=1
                else:
                    dicSpam[ng]=1
            
        else:    
            for word in tokens:
                word = stemmer.stem(word)
                stemmed.append(word)
 
            bigramsHam = ngrams(stemmed, 2)
            for ng in bigramsHam:
                if ng in dicHam:
                    dicHam[ng]+=1
                else:
                    dicHam[ng]=1                        



totalCountsHam = sum(dicHam.itervalues())
totalCountsSpam = sum(dicSpam.itervalues())



for ng in dicSpam:
    if dicSpam[ng] > ThOccurance:
        dictionary.append(ng)
        ngCountSpam[ng] = dicSpam[ng]

for ng in dicHam:
    if dicHam[ng] > ThOccurance:
        if ng not in dictionary:        
            dictionary.append(ng)
        ngCountHam[ng] = dicHam[ng]
        
 
V = len(dictionary)        
""" 
Laplace Smoothing       
"""     
for ng in dictionary:
    if ng in ngCountSpam:    
        ngCountSpam[ng] = float(ngCountSpam[ng]+1)/float(totalCountsSpam+V)
    else:
         ngCountSpam[ng] = (1.0)/float(totalCountsSpam+V)

    if ng in ngCountHam:    
        ngCountHam[word] = float(ngCountHam[ng]+1)/float(totalCountsHam+V)
    else:
         ngCountHam[ng] = (1.0)/float(totalCountsHam+V)



"""    
sorted_dictSpam = sorted(dicSpam.items(), key=operator.itemgetter(1), reverse = True)
sorted_dictHam = sorted(dicHam.items(), key=operator.itemgetter(1), reverse = True)
"""

nonNG = [("xxdollar","xxnumber"),( 'xxnumber', 'xxnumber'),("xxnumber","xxdollar")]
for ng in nonNG:
    del dictionary[dictionary.index(ng)]
    del ngCountHam[ng]
    del ngCountSpam[ng]


         

sorted_dictSpam = sorted(ngCountSpam.items(), key=operator.itemgetter(1), reverse = True)
sorted_dictHam = sorted(ngCountHam.items(), key=operator.itemgetter(1), reverse = True)


with open(projectPath+"5foldCV/"+str(NUMBER)+"/dictionaryBG"+str(ThOccurance)+".txt",'wu') as f:
    f.write('\n'.join('%s' % str(x) for x in dictionary))


with open(projectPath+"5foldCV/"+str(NUMBER)+"/bgCountHam"+str(ThOccurance)+".txt",'wu') as f:
    f.write('\n'.join('%s ' % str(x) for x in sorted_dictHam))
    
with open(projectPath+"5foldCV/"+str(NUMBER)+"/bgCountSpam"+str(ThOccurance)+".txt",'wu') as f:
    f.write('\n'.join('%s ' % str(x) for x in sorted_dictSpam))    

print "Dictionary Size: " + str(len(dictionary))


with open(projectPath+"/5foldCV/"+str(NUMBER)+"/dictionaryBG"+str(ThOccurance)+".pickle", 'wb') as handle:
  pickle.dump(dictionary, handle)
with open(projectPath+'/5foldCV/'+str(NUMBER)+"/bgCountHamDic"+str(ThOccurance)+".pickle", 'wb') as handle:
  pickle.dump(ngCountHam, handle)
with open(projectPath+"/5foldCV/"+str(NUMBER)+"/bgCountSpamDic"+str(ThOccurance)+".pickle", 'wb') as handle:
  pickle.dump(ngCountSpam, handle)
