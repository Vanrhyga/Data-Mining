#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*


import os
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier as SGD



mailList=[]
labels=[]
for i in os.listdir('./train/'):
    with open('./train/%s'%i,'r',encoding='gb18030',errors='ignore') as file:
        mail=file.read()
        #Using the jieba.analyse select the most significant words.
        mail=' '.join(jieba.analyse.extract_tags(mail,topK=20))
        mailList.append(mail)

with open('train_label.txt',encoding='utf8') as file:
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

#Regularization parameter.
sgdParams={'alpha:'[0.00006,0.00007,0.00008,0.0001,0.0005]}

#Find out which regularization parameter works the best.
modelSGD=GridSearchCV(SGD(random_state=0,shuffle=True),sgdParams,scoring='f1',cv=20)

#Fit the model
modelSGD.fit(trainSet,trainSetLabel)
SGDResult=modelSGD.predict_proba(devSet)[:,1]

print(modelSGD.best_score_)






