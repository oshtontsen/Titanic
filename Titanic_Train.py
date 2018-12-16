# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:42:52 2018

@author: Seizure
"""

# Code you have previously used to load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


#CHANGE path
#Path of the file to read. 
titanic_file_path ='C:/Users/Seizure/Documents/MachineLearning/Titanic/trainpostprocessing.csv'
titanic_data = pd.read_csv(titanic_file_path)
y= titanic_data.Survived

#CHANGE features
features = ['Age', 'Fare', 'IsMale', 'IsFemale', 'Is1stClass', 'Is2ndClass', 
            'Is3rdClass', 'EmbarkSouthampton', 'EmbarkCherbourg', 
            'EmbarkQueenstown', 'IsAlone']
X = titanic_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_pipeline = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators = 1000, n_early_stopping_rounds = 5, 
                                learning_rate = .05, n_jobs = 4))
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(val_X)
scores = cross_val_score(my_pipeline, X, y, cv = 10, scoring = 'neg_mean_absolute_error')
print("This is the accuracy of the model: " + str(scores))
print("Model accuracy: " + str(scores) + '% of the time.')
print("")
print("We are right: " + str(100*(1 + sum(scores)/len(scores))) + '% of the time.')

titanic_test_path = 'C:/Users/Seizure/Documents/MachineLearning/Titanic/testpostprocessing.csv'
titanic_test_data = pd.read_csv(titanic_test_path)
test_X = titanic_test_data[features]
test_predictions = my_pipeline.predict(test_X)
print(test_predictions)
df=pd.concat([titanic_test_data, pd.DataFrame(test_predictions)],axis=1)
df.to_csv('solution.csv', index=False)
