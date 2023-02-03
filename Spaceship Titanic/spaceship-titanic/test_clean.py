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

df = pd.read_csv("test.csv")

df.drop("PassengerId", axis = 1, inplace=True)


#HomePlanet column
rep = {"HomePlanet": {"Earth": 1, "Europa": 2, "Mars": 3},
"Destination": {"TRAPPIST-1e": 1, "55 Cancri e": 2, "PSO J318.5-22": 3},
}

df['CryoSleep'] = df['CryoSleep'].replace(True, 1)
df['VIP'] = df['VIP'].replace(True, 1)


df['CryoSleep'] = df['CryoSleep'].replace(False, 0)
df['VIP'] = df['VIP'].replace(False, 0)



df = df.replace(rep)

df["HomePlanet"] = df["HomePlanet"].fillna(0)
df["CryoSleep"] = df["CryoSleep"].fillna(0)
df["Destination"] = df["Destination"].fillna(0)
df["VIP"] = df["VIP"].fillna(0)



age_mean  = df["Age"].mean()
roomService_mean = df['RoomService'].mean()
FoodCourt_mean = df['FoodCourt'].mean()
Shopping_mean = df['ShoppingMall'].mean()
Spa_mean = df['Spa'].mean()
VRDeck_mean = df['VRDeck'].mean()

df['Age'] = df["Age"].fillna(age_mean)
df['RoomService'] = df['RoomService'].fillna(roomService_mean)
df['FoodCourt'] = df['FoodCourt'].fillna(FoodCourt_mean)
df['ShoppingMall'] = df['ShoppingMall'].fillna(Shopping_mean)
df['Spa'] = df['Spa'].fillna(Spa_mean)
df['VRDeck'] = df['VRDeck'].fillna(VRDeck_mean)

"""df['Name'] = df['Name'].str.extract(r' (\w+)')

val = df["Name"].value_counts()

df['Name'] = df['Name'].fillna('na')
df['Cabin'] = df['Cabin'].fillna('na')

from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
df["Family"] = ord_enc.fit_transform(df[["Name"]])


df["Cab"] = ord_enc.fit_transform(df[["Cabin"]])
"""
df[['Deck','Num','Side']] = df.Cabin.str.split('/',expand=True)

df.drop('Name', axis = 1, inplace = True)
df.drop('Cabin', axis = 1, inplace = True)

re = {"Deck": {'F':1, 'G':2, 'E': 3, 'B':4, 'C':5, 'D':6, 'A':7, 'T':8},
"Side": {'S':1, 'P':2}}

df = df.replace(re)

df['Deck'] = df["Deck"].fillna(0)
df['Num'] = df["Num"].fillna(0)
df['Side'] = df["Side"].fillna(0)

df.to_csv('newTest.csv')