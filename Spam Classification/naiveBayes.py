#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*



import os
from re import sub
from jieba import analyse
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


trainList=[]
testList=[]
idList=[]
content={}
trainSetLabel=[]

with open('trainLabel.txt',encoding='gb18030') as file:
    for line in file:
        line=line.split()
        content[line[1]]=line[0]

for i in os.listdir('./train/'):
    words=[]
    with open('./train/%s'%i,encoding='gb18030',errors='ignore') as file:
        mail=file.read()
    words=' '.join(analyse.extract_tags(mail,topK=20))
    trainList.append(words)

    i = sub('[..txt]', '', i)
    trainSetLabel.append(content[i])

for i in os.listdir('./test/'):
    words=[]
    with open('./test/%s'%i,encoding='gb18030',errors='ignore') as file:
        mail=file.read()
    words=' '.join(analyse.extract_tags(mail,topK=20))
    testList.append(words)

    i = sub('[..txt]', '', i)
    idList.append(i)

allData=trainList+testList
lenTrain=len(trainList)

ft=TFIV()
featureVector=ft.fit_transform(allData).toarray()

#Separate back into training and dev sets.
trainSet=featureVector[:lenTrain]
testSet=featureVector[lenTrain:]

#rain naive bayes model.
mnb=MultinomialNB()
MNBResult=mnb.fit(trainSet,trainSetLabel).predict(testSet)

outPut=pd.DataFrame(data={"classification":MNBResult,"id":idList,"content":testList})
outPut.to_excel("Result.xlsx",sheet_name='Result',index=False)