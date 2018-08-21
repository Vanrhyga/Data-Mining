#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*



import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def segment(line):
    return list(jieba.cut(line))

def processData(fileList):
    vectoring=TfidfVectorizer(input='content',tokenizer=segment,analyzer='word')

    content=[]
    counter=[]
    for fileName in fileList:
        beforeSize=len(content)
        content.extend(open(fileName).readlines())
        counter.append(len(content)-beforeSize)
    x=vectoring.fit_transform(content)
    y=np.concatenate((np.repeat([1],counter[0],axis=0),
                      np.repeat([0],counter[1],axis=0)),axis=0)
    return x,y,vectoring

def resultVectoring(v):
    v=v.reshape(-1,1)
    return np.concatenate((v*(-1)+1,v),axis=1)

def featureSelect(x,y):
    return SelectKBest(chi2,k=500).fit_transform(x,y)

x,y,model=processData(['Ham.txt','Spam.txt'])
x=featureSelect(x,y)
xTrain,xDev,yTrain,yDev=train_test_split(x,y,test_size=0.1)
model.classifier=LogisticRegression
tmp=model.classifier(class_weight='balanced')
model.clf=tmp.fit(xTrain,yTrain)
yPred=model.clf.predict(xDev)
yVec=resultVectoring(yDev)
print(metrics.classification_report(yDev,yPred))