import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.activations import relu, sigmoid, leaky_relu
from keras.layers import Dense, BatchNormalization, Normalization, InputLayer
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dropout
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('train.csv')


y = np.array(df['label'])
x = np.array(df.drop('label', axis = 1))


xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size= 0.8, test_size=0.2, random_state = 0)

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)

model = Sequential(
    [
        #InputLayer(input_shape=(xtrain.shape[1],)),
        BatchNormalization(),
        Dense(128, activation = 'relu', name = 'layer1', kernel_regularizer = 'l2'),
        Dropout(0.5),
        Dense(10, activation = 'softmax', name = 'layer4'),
    ]
)

model.compile(
    loss = CategoricalCrossentropy(),
    optimizer = Adam(learning_rate = 0.001),
    metrics = ['accuracy']
)

earlyStopping = EarlyStopping(monitor='val_loss', patience=6)
reduceLROnPlateau = ReduceLROnPlateau(patience=5)

model.fit(
    xtrain, ytrain,
    epochs = 1,
    validation_data = (xtest, ytest),
    verbose = 2,
    batch_size = 10,
    callbacks=[earlyStopping, reduceLROnPlateau]
    )

model.evaluate(xtest, ytest)

test = pd.read_csv('test.csv')
x_testn = np.array(test)

pred = model.predict(x_testn)

labels_pred = np.argmax(pred, axis=1)



df_R = pd.DataFrame(labels_pred, columns = ['Label'])

df_R.index = np.arange(1, len(df_R)+1)

df_R.index.name = ('ImageId')



#df_R['ImageId'] = np.arange(len(df_R))
print(df_R)



#df_R.reset_index(drop = True, inplace = True)

df_R.to_csv('submission.csv')