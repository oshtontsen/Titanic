# Code you have previously used to load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

#Path of the file to read. We changed the directory structure to show we use online IDEs like true intellectuals.
titanic_file_path = "C:/Users/OSHX1/Documents/Titanic-master/trainpostprocessing.csv"
titanic_data = pd.read_csv(titanic_file_path)
y= titanic_data.Survived

features = ['Age', 'Fare', 'IsMale', 'IsFemale', 'Is1stClass', 'Is2ndClass', 'Is3rdClass', 'EmbarkSouthampton', 'EmbarkCherbourg', 'EmbarkQueenstown', 'IsAlone']
X = titanic_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#for incre in range(0.03,0.05,0.01):
my_pipeline = make_pipeline(SimpleImputer(), XGBClassifier(n_estimators = 1000, n_early_stopping_rounds = 5, learning_rate = 0.05, n_jobs = 4))
my_pipeline.fit(train_X, train_y)
scores = cross_val_score(my_pipeline, X, y, cv=10,  scoring = "neg_mean_absolute_error")

#Make predictions
titanic_test_path = "C:/Users/OSHX1/Documents/Titanic-master/testpostprocessing.csv"
titanic_test_data = pd.read_csv(titanic_test_path)
test_X = titanic_test_data[features]
test_predictions = my_pipeline.predict(test_X)
print("These are the predictions: ")
print("")
print(test_predictions)

df = pd.concat([titanic_test_data, pd.DataFrame(test_predictions, columns=["Survived"])],axis=1)
#df = pd.DataFrame(test_predictions, columns=["Survived"])
df.to_csv('list3.csv', index=False)
