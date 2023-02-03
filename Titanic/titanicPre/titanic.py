import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.activations import relu, sigmoid, leaky_relu
from keras.layers import Dense, BatchNormalization, Normalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dropout
#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('train.csv')

#change column Name to title 
df['Titles'] = df['Name'].str.extract(r', (\w+\.)')
df.drop("Name", axis = 1, inplace = True)
df["Titles"] = df["Titles"].replace("Mr.", 1)
df["Titles"] = df["Titles"].replace("Mrs.", 2)
df["Titles"] = df["Titles"].replace("Miss.", 3)
df["Titles"] = df["Titles"].replace("Master.", 4)
df["Titles"] = df["Titles"].replace("Rev.", 5)
df["Titles"] = df["Titles"].replace("Dr.", 6)
df["Titles"] = df["Titles"].replace("Major.", 7)
df["Titles"] = df["Titles"].replace("Col.",8)
df["Titles"] = df["Titles"].replace("Don.",9)
df["Titles"] = df["Titles"].replace("Mme.",10)
df["Titles"] = df["Titles"].replace("Ms.",11)
df["Titles"] = df["Titles"].replace("Lady.", 12)
df["Titles"] = df["Titles"].replace("Sir.",13)
df["Titles"] = df["Titles"].replace("Mlle.",14)
df["Titles"] = df["Titles"].replace("Capt.",15)
df["Titles"] = df["Titles"].replace("Jonkheer.",16)

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

df.drop("PassengerId", axis = 1, inplace = True)


#leave only the alphabet part of the tickets
df['TicketTitles'] = df['Ticket'].str.extract(r'([A-Z]+)')
df.drop("Ticket", axis = 1, inplace = True)
df['TicketTitles'] = df['TicketTitles'].fillna('NA')

df["TicketTitles"] = df["TicketTitles"].replace("NA", 1)
df["TicketTitles"] = df["TicketTitles"].replace("PC", 2)
df["TicketTitles"] = df["TicketTitles"].replace("C", 3)
df["TicketTitles"] = df["TicketTitles"].replace("A", 4)
df["TicketTitles"] = df["TicketTitles"].replace("STON", 5)
df["TicketTitles"] = df["TicketTitles"].replace("SOTON", 6)
df["TicketTitles"] = df["TicketTitles"].replace("S", 7)
df["TicketTitles"] = df["TicketTitles"].replace("CA",8)
df["TicketTitles"] = df["TicketTitles"].replace("SC",9)
df["TicketTitles"] = df["TicketTitles"].replace("W",10)
df["TicketTitles"] = df["TicketTitles"].replace("F",11)
df["TicketTitles"] = df["TicketTitles"].replace("LINE", 12)
df["TicketTitles"] = df["TicketTitles"].replace("PP",13)
df["TicketTitles"] = df["TicketTitles"].replace("P",14)
df["TicketTitles"] = df["TicketTitles"].replace("WE",15)
df["TicketTitles"] = df["TicketTitles"].replace("SCO",16)
df["TicketTitles"] = df["TicketTitles"].replace("SW",17)
df["TicketTitles"] = df["TicketTitles"].replace("SO",18)

#dropping duplicates
df.drop_duplicates()
#replace all null values in cabin with 0 and all string values with 1

df['CabinTitles'] = df['Cabin'].str.extract(r'([A-Z]+)')
df.drop("Cabin", axis = 1, inplace = True)
df['CabinTitles'] = df['CabinTitles'].fillna('NA')

df["CabinTitles"] = df["CabinTitles"].replace("NA", 1)
df["CabinTitles"] = df["CabinTitles"].replace("C", 2)
df["CabinTitles"] = df["CabinTitles"].replace("B", 3)
df["CabinTitles"] = df["CabinTitles"].replace("D", 4)
df["CabinTitles"] = df["CabinTitles"].replace("E", 5)
df["CabinTitles"] = df["CabinTitles"].replace("A", 6)
df["CabinTitles"] = df["CabinTitles"].replace("F", 7)
df["CabinTitles"] = df["CabinTitles"].replace("G", 8)
df["CabinTitles"] = df["CabinTitles"].replace("T", 9)



#Adding the values of Parch and sibsp
added = df['SibSp']+df['Parch']
df['FamilyType'] = added
df.drop("SibSp", axis = 1, inplace = True)
df.drop("Parch", axis = 1, inplace = True)

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

print(df)
df.to_csv("newTrain.csv")
#spliting dataset into x and y
y = np.array(df["Survived"])

x = np.array(df.drop(["Survived"], axis = 1))

#splitting dataset into test and train datasets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state=0)

print("xtrain.shape: ", xtrain.shape, "ytrain.shape: ", ytrain.shape)
print("xtest.shape: ", xtest.shape, "ytest.shape: ", ytest.shape)

#get the test.csv values to test 
"""test = pd.read_csv('test.csv')

test.drop("PassengerId", axis = 1, inplace = True)


test['Titles'] = test['Name'].str.extract(r', (\w+\.)')
test.drop("Name", axis = 1, inplace = True)
test["Titles"] = test["Titles"].replace("Mr.", 1)
test["Titles"] = test["Titles"].replace("Mrs.", 2)
test["Titles"] = test["Titles"].replace("Miss.", 3)
test["Titles"] = test["Titles"].replace("Master.", 4)
test["Titles"] = test["Titles"].replace("Rev.", 5)
test["Titles"] = test["Titles"].replace("Dr.", 6)
test["Titles"] = test["Titles"].replace("Major.", 7)
test["Titles"] = test["Titles"].replace("Col.",8)

#Adding the values of Parch and sibsp
added = test['SibSp']+test['Parch']
test['FamilyType'] = added
test.drop("SibSp", axis = 1, inplace = True)
test.drop("Parch", axis = 1, inplace = True)

#replace all null values in cabin with 0 and all string values with 1
df['Cabin'] = df['Cabin'].fillna(0)
df['Cabin'] = pd.to_numeric(df['Cabin'], errors = 'coerce').fillna(1).astype(int)
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

#sklearn solution model with 82% accuracy
lr_model = LogisticRegression()
lr_model.fit(xtrain, ytrain)

y_pred = lr_model.predict(test_array)

prediction = np.transpose(y_pred)

dfLr = pd.DataFrame(prediction, columns = ["Prediction"])

dfLr.to_csv("LogisticResult.csv")

print("sklearn model accuracy on training set: ", lr_model.score(test_array, prediction)*100, "%")"""

#neural network implementation with 65% accuracy

#norm_l = Normalization(axis = -1)
#norm_l.adapt(xtrain)
#xn = norm_l(xtrain)

xt = np.tile(xtrain, (3000, 1))
yt = np.tile(ytrain,(1, 3000))
print(xt.shape)
yt = np.transpose(yt)
print(yt.shape)

model = Sequential(
    [
        BatchNormalization(),
        Dense(128, activation = 'relu', name = 'layer1', kernel_regularizer = 'l2'),
        Dropout(0.6),
        Dense(128, activation = 'relu', name = 'layer2', kernel_regularizer = 'l2'),
        Dropout(0.6),
        Dense(1, activation = 'sigmoid', name = 'layer3')
    ]
)

model.build(xt.shape)

model.summary()

model.compile(
    loss = BinaryCrossentropy(),
    optimizer = SGD(learning_rate = 0.0005),
    metrics = ['accuracy']
)

model.fit(
    xt, yt,
    epochs = 15,
    validation_data = (xtest, ytest),
    batch_size = 30,
    verbose = 2
)

loss, acc = model.evaluate(xtest, ytest)
print("loss: ", loss,"\nAccuracy: ", acc)

test = pd.read_csv('test.csv')

test.drop("PassengerId", axis = 1, inplace = True)

#leave only the alphabet part of the tickets
test['TicketTitles'] = test['Ticket'].str.extract(r'([A-Z]+)')
test.drop("Ticket", axis = 1, inplace = True)
test['TicketTitles'] = test['TicketTitles'].fillna('NA')

test["TicketTitles"] = test["TicketTitles"].replace("NA", 1)
test["TicketTitles"] = test["TicketTitles"].replace("PC", 2)
test["TicketTitles"] = test["TicketTitles"].replace("C", 3)
test["TicketTitles"] = test["TicketTitles"].replace("A", 4)
test["TicketTitles"] = test["TicketTitles"].replace("STON", 5)
test["TicketTitles"] = test["TicketTitles"].replace("SOTON", 6)
test["TicketTitles"] = test["TicketTitles"].replace("S", 7)
test["TicketTitles"] = test["TicketTitles"].replace("CA",8)
test["TicketTitles"] = test["TicketTitles"].replace("SC",9)
test["TicketTitles"] = test["TicketTitles"].replace("W",10)
test["TicketTitles"] = test["TicketTitles"].replace("F",11)
test["TicketTitles"] = test["TicketTitles"].replace("LINE", 12)
test["TicketTitles"] = test["TicketTitles"].replace("PP",13)
test["TicketTitles"] = test["TicketTitles"].replace("P",14)
test["TicketTitles"] = test["TicketTitles"].replace("WE",15)
test["TicketTitles"] = test["TicketTitles"].replace("SCO",16)
test["TicketTitles"] = test["TicketTitles"].replace("SW",17)
test["TicketTitles"] = test["TicketTitles"].replace("SO",18)
test["TicketTitles"] = test["TicketTitles"].replace("AQ",19)
test["TicketTitles"] = test["TicketTitles"].replace("LP",20)

test['Titles'] = test['Name'].str.extract(r', (\w+\.)')
test.drop("Name", axis = 1, inplace = True)
test["Titles"] = test["Titles"].replace("Mr.", 1)
test["Titles"] = test["Titles"].replace("Mrs.", 2)
test["Titles"] = test["Titles"].replace("Miss.", 3)
test["Titles"] = test["Titles"].replace("Master.", 4)
test["Titles"] = test["Titles"].replace("Rev.", 5)
test["Titles"] = test["Titles"].replace("Dr.", 6)
test["Titles"] = test["Titles"].replace("Major.", 7)
test["Titles"] = test["Titles"].replace("Col.",8)
test["Titles"] = test["Titles"].replace("Don.",9)
test["Titles"] = test["Titles"].replace("Mme.",10)
test["Titles"] = test["Titles"].replace("Ms.",11)
test["Titles"] = test["Titles"].replace("Lady.", 12)
test["Titles"] = test["Titles"].replace("Sir.",13)
test["Titles"] = test["Titles"].replace("Mlle.",14)
test["Titles"] = test["Titles"].replace("Capt.",15)
test["Titles"] = test["Titles"].replace("Jonkheer.",16)
test["Titles"] = test["Titles"].replace("Dona.", 17)

#Adding the values of Parch and sibsp
added = test['SibSp']+test['Parch']
test['FamilyType'] = added.astype(int)
test.drop("SibSp", axis = 1, inplace = True)
test.drop("Parch", axis = 1, inplace = True)

#replace all null values in cabin with 0 and all string values with 1
test['CabinTitles'] = test['Cabin'].str.extract(r'([A-Z]+)')
test.drop("Cabin", axis = 1, inplace = True)
test['CabinTitles'] = test['CabinTitles'].fillna('NA')

test["CabinTitles"] = test["CabinTitles"].replace("NA", 1)
test["CabinTitles"] = test["CabinTitles"].replace("C", 2)
test["CabinTitles"] = test["CabinTitles"].replace("B", 3)
test["CabinTitles"] = test["CabinTitles"].replace("D", 4)
test["CabinTitles"] = test["CabinTitles"].replace("E", 5)
test["CabinTitles"] = test["CabinTitles"].replace("A", 6)
test["CabinTitles"] = test["CabinTitles"].replace("F", 7)
test["CabinTitles"] = test["CabinTitles"].replace("G", 8)
test["CabinTitles"] = test["CabinTitles"].replace("T", 9)
 
test_one = pd.get_dummies(test["Sex"])
test_two= pd.concat((test_one, test), axis = 1)
test_two.drop(["Sex", "male"], axis = 1, inplace = True)
test_two.rename(columns={'female': 'Sex'}, inplace = True)

test = test_two
test["Embarked"] = test["Embarked"].replace("C", 1)
test["Embarked"] = test["Embarked"].replace("S", 2)
test["Embarked"] = test["Embarked"].replace("Q", 3)
#replacing all the null values in column Age with the mean age
age_mean  = test["Age"].mean()

test['Age'] = test["Age"].fillna(age_mean)
test.drop_duplicates()

test = test.fillna(0)

#test = test.dropna(axis = 0, how="any")

test.to_csv("newTest.csv")

x_testn = np.array(test)
m, n = x_testn.shape
#x_testn = norm_l(x_testn)

yhat = np.zeros([m, 1], dtype=int)

a1 = model.predict(x_testn)

for i in range(len(a1)):
    if a1[i] < 0.5:
        yhat[i] = 0
    elif a1[i] >= 0.5:
        yhat[i] = 1

result = np.concatenate((x_testn, yhat), axis = 1)

df_R = pd.DataFrame(result, columns = ["Sex", "Pclass", "Age", "Fare", "Embarked",  "TicketTitles", "Titles", "FamilyType", "CabinTitles", "Prediction"])

print(df_R)

df_R.to_csv("result.csv")