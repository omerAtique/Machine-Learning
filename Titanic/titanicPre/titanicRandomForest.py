import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.activations import relu, sigmoid, leaky_relu
from keras.layers import Dense, Normalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from keras.models import Sequential
import pandas as pd
from tensorflow import decision_forests as tfdf

df = pd.read_csv('train.csv')

#Drop column Name as name is not a good feature to train model on
df.drop("Name", axis = 1, inplace = True)

#Assigning binary values to values in Column Sex
df_one = pd.get_dummies(df["Sex"])
df_two= pd.concat((df_one, df), axis = 1)
df_two.drop(["Sex", "male"], axis = 1, inplace = True)
df_two.rename(columns={'female': 'Sex'}, inplace = True)

df = df_two

#seperating numeric and non numeric data
df_numeric = df.select_dtypes(include = [np.number])
numeric_cols = df_numeric.columns.values

df_non_numeric = df.select_dtypes(exclude = [np.number])
non_numeric_cols  = df_non_numeric.columns.values

#dropping column Cabin as it has many null values
df.drop("Cabin", axis = 1, inplace = True)
df.drop("Ticket", axis = 1, inplace = True)
df.drop("PassengerId", axis = 1, inplace = True)
#dropping duplicates
df.drop_duplicates()

#locating and dropping all null values
df = df.dropna(axis = 0, how="any")

#converting embarked into a categorical feature
"""df_E = pd.get_dummies(df["Embarked"])
df_E = pd.concat((df_E, df), axis = 1)
df_E.drop(["Embarked"], axis = 1, inplace = True)
df = df_E"""

df["Embarked"] = df["Embarked"].replace("C", 1)
df["Embarked"] = df["Embarked"].replace("S", 2)
df["Embarked"] = df["Embarked"].replace("Q", 3)
#replacing all the null values in column Age with the mean age
age_mean  = df["Age"].mean()

df['Age'] = df["Age"].fillna(age_mean)

#spliting dataset into x and y
y = np.array(df["Survived"])

x = np.array(df.drop(["Survived"], axis = 1))

#splitting dataset into test and train datasets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state=0)

print("xtrain.shape: ", xtrain.shape, "ytrain.shape: ", ytrain.shape)
print("xtest.shape: ", xtest.shape, "ytest.shape: ", ytest.shape)

#get the test.csv values to test 
test = pd.read_csv('test.csv')

test.drop("PassengerId", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)

test_one = pd.get_dummies(test["Sex"])
test_two= pd.concat((test_one, test), axis = 1)
test_two.drop(["Sex", "male"], axis = 1, inplace = True)
test_two.rename(columns={'female': 'Sex'}, inplace = True)

test = test_two
test["Embarked"] = test["Embarked"].replace("C", 1)
test["Embarked"] = test["Embarked"].replace("S", 2)
test["Embarked"] = test["Embarked"].replace("Q", 3)

age_mean  = test["Age"].mean()

test['Age'] = test["Age"].fillna(age_mean)
test.drop_duplicates()
test = test.dropna(axis = 0, how="any")

test_array = np.array(test)

#Random Forest implementation

