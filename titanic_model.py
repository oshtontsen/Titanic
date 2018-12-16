# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

#CHANGE path
#Path of the file to read. We changed the directory structure to show we use online IDEs like true intellectuals.
titanic_file_path = “googledrive/train.csv”
titanic_data = pd.read_csv(titanic_file_path)
y= titanic_data.Survived

#CHANGE features
features = ['Age', 'Fare', 'IsMale', 'IsFemale', 'Is1stClass', 'Is2ndClass', 'Is3rdClass', 'EmbarkSouthhampton', 'EmbarkCherbourg', 'EmbarkQueenstown', 'IsAlone']
X = titanic_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators = 1000, n_early_stopping_rounds = 5, n_jobs = 4))
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
scores = cross_val_score(my_pipeline, X, y, scoring = “neg_mean_absolute_error”)
print(‘This is the accuracy of the model: ‘ + scores)
