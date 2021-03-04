# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 00:43:01 2020

@author: ishaa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv("C:\\Users\\ishaa\\OneDrive\\Desktop\\titanic\\train.csv")
test=pd.read_csv("C:\\Users\\ishaa\\OneDrive\\Desktop\\titanic\\test.csv")
sample_submission=pd.read_csv("C:\\Users\\ishaa\\OneDrive\\Desktop\\titanic\\gender_submission.csv")

a=train.describe(include='all')

age=pd.DataFrame(data=train[['Age','Survived']])
age['age_bins']=pd.cut(x=age['Age'],bins=[0,17,100])



sns.barplot(x='age_bins', y="Survived",data=age)
age['age_bins']=age['age_bins'].apply(str)
print("Percentage of children who survived:",age["Survived"][age['age_bins']=='(0,17]' ].value_counts(normalize=True)[1]*100)
print("Percentage of adults who survived:",age["Survived"][age['age_bins']=='(17,100]'].value_counts(normalize=True)[1]*100)

train.info()
train1=train.drop('Cabin',axis=1)
train1=train1.drop('Ticket',axis=1)
train1=train1.drop('Name',axis=1)
train1=train1.drop('PassengerId',axis=1)

corr=train1.corr()
sns.heatmap(corr,annot=True)

from scipy import stats
F,p = stats.f_oneway(train1[train1.Sex=='male'].Survived,train1[train1.Sex=='female'].Survived)
print(F)

from scipy import stats
F, p = stats.f_oneway(train1[train1.Embarked=='C'].Survived,train1[train1.Embarked=='S'].Survived,train1[train1.Embarked=='Q'].Survived)
print(F)

train1['Pclass']=train1['Pclass'].apply(str)
from scipy import stats
F,p = stats.f_oneway(train1[train1.Pclass=='1'].Survived,train1[train1.Pclass=='2'].Survived,train1[train1.Pclass=='3'].Survived)
print(F)

