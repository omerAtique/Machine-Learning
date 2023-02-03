import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import tensorflow as tf
from keras.activations import relu, sigmoid, leaky_relu
from keras.layers import Dense, BatchNormalization, Normalization
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dropout
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

exec(open("trainEng.py").read())
exec(open("testEng.py").read())

df = pd.read_csv('newTrain.csv')

y = np.array(df['SalePrice'])
x = np.array(df.drop(['SalePrice'], axis = 1))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, test_size = 0.3, random_state=0)

print("xtrain.shape: ", xtrain.shape, "ytrain.shape: ", ytrain.shape)
print("xtest.shape: ", xtest.shape, "ytest.shape: ", ytest.shape)

test = pd.read_csv('newTest.csv')
test_array = np.array(test)


#Decision Tree model
model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(xtrain, ytrain)

pred = model.predict(test_array)

res = pd.read_csv('test.csv')

id = res['Id']

Id = np.array(id)

pred.shape = (pred.shape[0], 1)
Id.shape = (Id.shape[0], 1)

result = np.concatenate((Id, pred), axis = 1)

df_R = pd.DataFrame(result, columns = ['Id', 'SalePrice'])

df_R.to_csv('result2.csv')

varify = pd.read_csv('sample_submission.csv')

saleprice = varify['SalePrice']

pred = np.array(pred)
sale_price = np.array(saleprice)


print('Mean Absolute Error:', mean_absolute_error(sale_price, pred))  
print('Mean Squared Error:', mean_squared_error(sale_price, pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(sale_price, pred)))