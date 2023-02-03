import string
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

exec(open("train_clean.py").read())
exec(open("test_clean.py").read())

df = pd.read_csv('newTrain.csv')

y = np.array(df['Transported'])
x = np.array(df.drop(['Transported'], axis = 1))

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, test_size = 0.3, random_state=0)

print("xtrain.shape: ", xtrain.shape, "ytrain.shape: ", ytrain.shape)
print("xtest.shape: ", xtest.shape, "ytest.shape: ", ytest.shape)

test = pd.read_csv('newTest.csv')
x_testn = np.array(test)

"""
xt = np.tile(xtrain, (200, 1))
yt = np.tile(ytrain,(1, 200))
print(xt.shape)
yt = np.transpose(yt)
print(yt.shape)
xtrain = xt
ytrain = yt



norm_l = Normalization(axis = -1)
norm_l.adapt(xtrain)
xn = norm_l(xtrain)
"""
model = Sequential(
    [
        BatchNormalization(),
        Dense(64, activation = 'relu', name = 'layer1', kernel_regularizer = 'l2'),
        Dropout(0.5),
        Dense(1, activation = 'sigmoid', name = 'layer3')
    ]
)

model.build(xtrain.shape)

model.summary()

model.compile(
    loss = BinaryCrossentropy(),
    optimizer = SGD(learning_rate = 0.02),
    metrics = ['accuracy']
)

model.fit(
    xtrain, ytrain,
    epochs = 20,
    validation_data = (xtest, ytest),
    verbose = 2,
    batch_size=15
)

loss, acc = model.evaluate(xtest, ytest)
print("loss: ", loss,"\nAccuracy: ", acc)

m, n = x_testn.shape
#x_test = norm_l(x_testn)

yhat = np.zeros([m, 1], dtype=bool)

a1 = model.predict(x_testn)

for i in range(len(a1)):
    if a1[i] < 0.5:
        yhat[i] = False
    elif a1[i] >= 0.5:
        yhat[i] = True

res = pd.read_csv('test.csv')

id = res['PassengerId']

Id = np.array(id)

yhat.shape = (yhat.shape[0], 1)
Id.shape = (Id.shape[0], 1)

result = np.concatenate((Id, yhat), axis = 1)

df_R = pd.DataFrame(result, columns = ['PassengerId', 'Transported'])

#df_R.reset_index(drop = True, inplace = True)

df_R.to_csv('submission.csv', index = False)