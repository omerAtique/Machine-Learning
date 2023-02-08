import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('archive/data.csv')

x = heart_data.drop(columns = 'condition', axis = 1)
y = heart_data['condition']

print(x)
print(y)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)

model = LogisticRegression()

model.fit(x, y)

x_train_prediction = model.predict(xtrain)
training_data_accuracy = accuracy_score(x_train_prediction, ytrain)

print("Accuracy on training data: ", training_data_accuracy)


x_test_prediction = model.predict(xtest)
test_data_accuracy = accuracy_score(x_test_prediction, ytest)

print("Accuracy on test set: ", test_data_accuracy)

input1 = np.array([62,0,0,140,268,0,0,160,0,3.6,0,2,2])

input1_reshaped = input1.reshape(1, -1)

prediction = model.predict(input1_reshaped)

print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')



  input1 = np.array([89,1,0,160,298,0,0,190,0,4.6,0,2,2])

input1_reshaped = input1.reshape(1, -1)

prediction = model.predict(input1_reshaped)

print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')