# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:26:22 2018

@author: kaushik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import lime
import lime.lime_tabular


dataset = pd.read_csv("C://git//CustomerChurn_Bank//bank_churn.csv")

x = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

print(x.isnull().sum())
print(y.isnull().sum())

x = pd.get_dummies(x, prefix = ['Geography', 'Gender'], columns = ['Geography', 'Gender'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Initializing Neural Network
classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 13))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting our model 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)
#0.83, 0.85