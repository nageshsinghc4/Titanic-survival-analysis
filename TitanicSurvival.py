#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:07:03 2019

@author: nageshsinghchauhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy
import seaborn as sn
import os

import warnings ## importing warnings library. 
warnings.filterwarnings('ignore') ## Ignore warning

print(os.listdir("/Users/nageshsinghchauhan/Downloads/ML/kaggle/"))

#load training and test datasets
train = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/kaggle/train.csv")
test = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/kaggle/test.csv")

## saving passenger id in advance in order to submit later. 
passengerID = test.PassengerId

## We will drop PassengerID and Ticket since it will be useless for us
train.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Ticket','Name', 'Cabin'], axis = 1, inplace = True)
train.info()
test.info()

#Dealing with missing values
#Finding total and percentage of missing values in each column
def missingValue_Function(mv):
    totalMV = mv.isnull().sum().sort_values(ascending = False)
    percent = round(mv.isnull().sum().sort_values(ascending = False)/len(mv)*100, 2)
    return pd.concat([totalMV, percent], axis = 1, keys = ['Total', 'Percentage'])

print(missingValue_Function(train)) #missing values in training data
print(missingValue_Function(test)) #missing values in test data

#We see that in both train, and test dataset have missing values. 
#Let's make an effort to fill these missing values starting with "Embarked" feature.

def percent_value_counts(df, feature):
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    return pd.concat([total, percent], axis = 1, keys = ['Total', 'Percentage'])

print(percent_value_counts(train, 'Embarked')) #missing values in training data
#we have 2 Null values in Embarked column, lets check the records containg null values
train[train.Embarked.isnull()]

#We may be able to solve these two missing values by looking at other independent variables of the two raws. 
#Both passengers paid a fare of $80, are of Pclass 1 and female Sex. 
#Let's see how the Fare is distributed among all Pclass and Embarked feature values
import seaborn as sns
fig, ax = plt.subplots(figsize=(16,12),ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax = ax[0]);
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 18)
ax2.set_title('Test Set',  fontsize = 18)
fig.show()
#Here, in both training set and test set, the average fare closest to $80 are in the C Embarked values. So, let's fill in the missing values as "C"
train.Embarked.fillna("C", inplace = True)

 #fill missing values in Embarked column with C
missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
## replace the test.fare null values with test.fare mean
test.Fare.fillna(missing_value, inplace=True)

## Concat train and test into a variable "all_data"
survivers = train.Survived
all_data = pd.concat([train,test], ignore_index=False)
"""

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]

X_test  = test.copy()
X_train.shape, Y_train.shape, X_test.shape

"""


X = all_data.iloc[:,:-1]
y = all_data.iloc[:,7]


#taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,0])
X[:,:1] = imputer.transform(X[:,:1])

imputer1 = imputer.fit(y.reshape(-1,1))
y = imputer1.transform(y.reshape(-1,1))



#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X)