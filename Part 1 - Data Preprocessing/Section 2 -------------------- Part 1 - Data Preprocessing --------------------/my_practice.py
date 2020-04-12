# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:18:04 2020

@author: Ismail
"""


#importing libraries

import numpy as np # mathematical calculations
import matplotlib.pyplot as plt # to plot graphs
import pandas as pd # for handling datasets

#reading dataset

dataset = pd.read_csv('Data.csv')


# separating dependent and independant variables


X = dataset.iloc[:,:-1].values


Y = dataset.iloc[:,-1].values


# missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])



# encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencode = OneHotEncoder(categorical_features = [0])
X= onehotencode.fit_transform(X).toarray()



labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)







# splitting training and testing data


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

# feature scaling


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)