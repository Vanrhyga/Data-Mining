#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*



import os
import re


ham=''
spam=''
labels={}

with open('trainLabel.txt',encoding='gb18030') as file:
    for line in file:
        line=line.split()
        labels[line[1]]=line[0]

for i in os.listdir('./train/'):
    with open('./train/%s'%i,encoding='gb18030',errors='ignore') as file:
        mail=file.read()
        i=re.sub('[..txt]','',i)

        if labels[i]=='0':
            ham+=mail
            ham+='\n'
        else:
            spam+=mail
            spam+='\n'

outPut=open('Ham.txt','w')
outPut.write(ham)
outPut.close()

outPut=open('Spam.txt','w')
outPut.write(spam)
outPut.close()