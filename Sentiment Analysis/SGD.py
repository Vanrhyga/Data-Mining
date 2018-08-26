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
data1=pd.read_excel('trainSet1.xlsx',header=1,skiprows=0)
data2=pd.read_excel('trainSet2.xlsx',header=1,skiprows=0)

trainSet=data1.append(data2)

testSet=pd.read_csv('test.csv',header=0,delimiter='\t',quoting=3)

trainSet=trainSet.reset_index(drop=True)
testSet=testSet.reset_index(drop=True)

trainData=[]
for i in range(0,len(trainSet['review'])):
    trainData.append(" ".join(reviewToWordlist(trainSet['review'][i])))
testData=[]
for i in range(0,len(testSet['review'])):
    testData.append(" ".join(reviewToWordlist(testSet['review'][i])))

tfv=TFIV(min_df=3,max_features=None,
         strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
         ngram_range=(1,2),use_idf=1,smooth_idf=1,sublinear_tf=1,
         stop_words='english')

#Combine both to fit the TFIDF vectorization.
allData=trainData+testData
lenTrain=len(trainData)
tfv.fit(allData)
allData=tfv.transform(allData)

#Separate back into training and dev sets.
train=allData[:lenTrain]
test=allData[lenTrain:]

#Regularization parameter
sgdParams={'alpha':[0.00006,0.00007,0.00008,0.0001,0.0005]}

#Find out which regularization parameter works the best.
modelSGD=GridSearchCV(SGD(random_state=0,shuffle=True,loss='modified_huber'),
                      sgdParams,scoring='roc_auc',cv=20)

#Fit the model.
modelSGD.fit(train,trainSet['sentiment'])
SGDResult=modelSGD.predict_proba(test)[:,1]
SGDOutput=pd.DataFrame(data={"id":testSet['id'],"review":testSet['review'],"sentiment":SGDResult})
SGDOutput.to_excel("Result.xlsx",sheet_name='Result',index=False)
print(modelSGD.best_score_)
print(modelSGD.best_params_)