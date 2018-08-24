#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*a



import os
import gzip
import numpy as np
import tensorflow as tf
from keras.layers import Input,Lambda,Dropout,Dense
from keras import backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from scipy import misc
import tqdm


def reader(path,kind):
    labelsPath=os.path.join(path,
                            '%s-labels-idx1-ubyte.gz'
                            %kind)
    imagesPath=os.path.join(path,
                            '%s-images-idx3-ubyte.gz'
                            %kind)

    with gzip.open(labelsPath,'rb') as lb:
        labels=np.frombuffer(lb.read(),dtype=np.uint8,
                             offset=8)

    with gzip.open(imagesPath,'rb') as img:
        images=np.frombuffer(img.read(),dtype=np.uint8,
                             offset=16).reshape((len(labels)),784)

    return images,labels


def randomReverse(x):
    if np.random.random()>0.5:
        return x[:,::-1]
    else:
        return x


def dataGenerator(x,y,batchSize=100):
    while True:
        idxs=np.random.permutation(len(x))
        x=x[idxs]
        y=y[idxs]
        p,q=[],[]

        for i in range(len(x)):
            p.append(randomReverse(x[i]))
            q.append(y[i])

            if len(p)==batchSize:
                yield np.array(p),np.array(q)
                p,q=[],[]

        if p:
            yield np.array(p),np.array(q)
            p,q=[],[]

np.random.seed(2018)
tf.set_random_seed(2018)

xTrain,yTrain=reader('./mnist',kind='train')
xTest,yTest=reader('./mnist',kind='t10k')

height,width=56,56

inputImages=Input(shape=(height,width))
inputImages_=Lambda(lambda x:K.repeat_elements(K.expand_dims(x,3),3,3))(inputImages)

baseModel=MobileNet(input_tensor=inputImages_,include_top=False,pooling='avg')

output=Dropout(0.5)(baseModel.output)
predict=Dense(10,activation='softmax')(output)

model=Model(inputs=inputImages,outputs=predict)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

xTrain=xTrain.reshape((-1,28,28))
xTrain=np.array([misc.imresize(x,(height,width)).astype(float) for x in tqdm.tqdm(iter(xTrain))])/255.

xTest=xTest.reshape((-1,28,28))
xTest=np.array([misc.imresize(x,(height,width)).astype(float) for x in tqdm.tqdm(iter(xTest))])/255.

model.fit_generator(dataGenerator(xTrain,yTrain),steps_per_epoch=600,epochs=50,validation_data=dataGenerator(xTest,yTest),validation_steps=100)
