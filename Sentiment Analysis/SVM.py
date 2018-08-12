#!/usr/bin/env python3
# -*- coding: utf-8 -*


import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
from sklearn.linear_model import SGDClassifier as SGD



def reviewToWordlist(review):
    #First remove the HTML.
    reviewText=BeautifulSoup(review,features="html.parser").get_text()

    #Use regular expressions to only include words.
    reviewText=re.sub("[^a-zA-Z]"," ",reviewText)

    #Convert words to lower case and split them into separate words.
    words=reviewText.lower().split()

    #Return a list of words
    return words


#Import the data.
data1=pd.read_csv('trainSet1.csv',header=0,delimiter='\t',quoting=3)
data2=pd.read_csv('trainSet2.csv',header=0,delimiter='\n',quoting=3)

data=data1.append(data2)

print(data)

trainSet,devSet= train_test_split(data, test_size=0.1, random_state=0)

trainSet=trainSet.reset_index(drop=True)
devSet=devSet.reset_index(drop=True)

trainData=[]
for i in range(0,len(trainSet['review'])):
    trainData.append(" ".join(reviewToWordlist(trainSet['review'][i])))
devData=[]
for i in range(0,len(devSet['review'])):
    devData.append(" ".join(reviewToWordlist(devSet['review'][i])))

tfv=TFIV(min_df=3,max_features=None,
         strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
         ngram_range=(1,2),use_idf=1,smooth_idf=1,sublinear_tf=1,
         stop_words='english')

#Combine both to fit the TFIDF vectorization.
allData=trainData+devData
lenTrain=len(trainData)
tfv.fit(allData)
allData=tfv.transform(allData)

#Separate back into training and dev sets.
train=allData[:lenTrain]
dev=allData[lenTrain:]

#Regularization parameter
sgdParams={'alpha':[0.00006,0.00007,0.00008,0.0001,0.0005]}

#Find out which regularization parameter works the best.
modelSGD=GridSearchCV(SGD(random_state=0,shuffle=True,loss='modified_huber'),
                      sgdParams,scoring='roc_auc',cv=20)

#Fit the model.
modelSGD.fit(train,trainSet['sentiment'])
SGDResult=modelSGD.predict_proba(dev)[:,1]
SGDOutput=pd.DataFrame(data={"review":devSet['review'],"sentiment":SGDResult})
SGDOutput.to_csv("Result.csv",index=False,quoting=3)
print(modelSGD.best_score_)