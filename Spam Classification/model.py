#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*


import os
from re import sub,split
from jieba import cut,analyse
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



stopWords=[]
mailList=[]
labels=[]
with open('stopWords.txt',encoding='gb18030') as file:
    for line in file:
        line=line.strip()
        stopWords.append(line)

for i in os.listdir('./train/'):
    words=[]
    with open('./train/%s'%i,encoding='gb18030',errors='ignore') as file:
        for line in file:
            line=sub(r'[.【】0-9、——。，！~\*]','',line)
            line=cut(line)
            outStr=''
            for word in line:
                if word not in stopWords and len(word)>1 and word!='\t':
                    outStr+=word
        words=' '.join(analyse.extract_tags(outStr,topK=100))
    mailList.append(words)

with open('trainLabel.txt',encoding='utf8') as file:
    for line in file:
        line=line.split()
        labels.append(line[0])

#feature extraction
ft=TFIV()
featureVector=ft.fit_transform(mailList).toarray()

#Separate back into training and dev sets.
trainSet=featureVector[1000:-1000]
trainSetLabel=labels[1000:-1000]
devSet=np.append(featureVector[:1000],featureVector[-1000:],axis=0)
devSetLabel=np.append(labels[:1000],labels[-1000:],axis=0)

#rain naive bayes model.
mnb=MultinomialNB()
MNBResult=mnb.fit(trainSet,trainSetLabel).predict(devSet)

#F1-score
f1Score = metrics.classification_report(MNBResult, devSetLabel)
print(f1Score)






