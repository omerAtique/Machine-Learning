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

#Dropping ID
df.drop("Id", axis = 1, inplace = True)

#Droppin Alley as too many NA vals
df.drop("Alley", axis = 1, inplace = True)

#fill NA vals with 0 in LotFrontage
df['LotFrontage'] = df['LotFrontage'].replace('NA', 0)
df['LotFrontage'] = df['LotFrontage'].fillna(0)


#Turn Steet into a binary column
df['Street'] = df['Street'].replace('Grvl', 1)
df['Street'] = df['Street'].replace('Pave', 2)

#Replace non numeric with numeric val in MSZoning
df['MSZoning'] = df['MSZoning'].replace('A', 1)
df['MSZoning'] = df['MSZoning'].replace('C', 2)
df['MSZoning'] = df['MSZoning'].replace('FV', 3)
df['MSZoning'] = df['MSZoning'].replace('I', 4)
df['MSZoning'] = df['MSZoning'].replace('RH', 5)
df['MSZoning'] = df['MSZoning'].replace('RL', 6)
df['MSZoning'] = df['MSZoning'].replace('RP', 7)
df['MSZoning'] = df['MSZoning'].replace('RM', 8)
df['MSZoning'] = df['MSZoning'].replace('C (all)', 9)

#Replace non numeric with numeric val in lotShape
df['LotShape'] = df['LotShape'].replace('Reg', 1)
df['LotShape'] = df['LotShape'].replace('IR1', 2)
df['LotShape'] = df['LotShape'].replace('IR2', 3)
df['LotShape'] = df['LotShape'].replace('IR3', 4)

#Replace non numeric with numeric val in LandContour
df['LandContour'] = df['LandContour'].replace('Lvl', 1)
df['LandContour'] = df['LandContour'].replace('Bnk', 2)
df['LandContour'] = df['LandContour'].replace('HLS', 3)
df['LandContour'] = df['LandContour'].replace('Low', 4)

#Replace non numeric with numeric in Utilitites
df['Utilities'] = df['Utilities'].replace('AllPub', 1)
df['Utilities'] = df['Utilities'].replace('NoSewr', 2)
df['Utilities'] = df['Utilities'].replace('NoSeWa', 3)
df['Utilities'] = df['Utilities'].replace('ELO', 4)

#Replace non numeric with numeric in LotConfig
df['LotConfig'] = df['LotConfig'].replace('Inside', 1)
df['LotConfig'] = df['LotConfig'].replace('Corner', 2)
df['LotConfig'] = df['LotConfig'].replace('CulDSac', 3)
df['LotConfig'] = df['LotConfig'].replace('FR2', 4)
df['LotConfig'] = df['LotConfig'].replace('FR3', 5)

#Replace non numeric with numeric in LandSlope
df['LandSlope'] = df['LandSlope'].replace('Gtl', 1)
df['LandSlope'] = df['LandSlope'].replace('Mod', 2)
df['LandSlope'] = df['LandSlope'].replace('Sev', 3)

#Replace non numeric with numeric in Neighborhood
df['Neighborhood'] = df['Neighborhood'].replace('Blmngtn', 1)
df['Neighborhood'] = df['Neighborhood'].replace('Blueste', 2)
df['Neighborhood'] = df['Neighborhood'].replace('BrDale', 3)
df['Neighborhood'] = df['Neighborhood'].replace('BrkSide', 4)
df['Neighborhood'] = df['Neighborhood'].replace('ClearCr', 5)
df['Neighborhood'] = df['Neighborhood'].replace('CollgCr', 6)
df['Neighborhood'] = df['Neighborhood'].replace('Crawfor', 7)
df['Neighborhood'] = df['Neighborhood'].replace('Edwards', 8)
df['Neighborhood'] = df['Neighborhood'].replace('Gilbert', 9)
df['Neighborhood'] = df['Neighborhood'].replace('IDOTRR', 10)
df['Neighborhood'] = df['Neighborhood'].replace('MeadowV', 11)
df['Neighborhood'] = df['Neighborhood'].replace('Mitchel', 12)
df['Neighborhood'] = df['Neighborhood'].replace('Names', 13)
df['Neighborhood'] = df['Neighborhood'].replace('NoRidge', 14)
df['Neighborhood'] = df['Neighborhood'].replace('NPkVill', 15)
df['Neighborhood'] = df['Neighborhood'].replace('NridgHt', 16)
df['Neighborhood'] = df['Neighborhood'].replace('NWAmes', 17)
df['Neighborhood'] = df['Neighborhood'].replace('OldTown', 18)
df['Neighborhood'] = df['Neighborhood'].replace('SWISU', 19)
df['Neighborhood'] = df['Neighborhood'].replace('Sawyer', 20)
df['Neighborhood'] = df['Neighborhood'].replace('SawyerW', 21)
df['Neighborhood'] = df['Neighborhood'].replace('Somerst', 22)
df['Neighborhood'] = df['Neighborhood'].replace('StoneBr', 23)
df['Neighborhood'] = df['Neighborhood'].replace('Timber', 24)
df['Neighborhood'] = df['Neighborhood'].replace('Veenker', 25)
df['Neighborhood'] = df['Neighborhood'].replace('NAmes', 26)

#Replace non numeric with numeric in Condition1
df['Condition1'] = df['Condition1'].replace('Artery', 1)
df['Condition1'] = df['Condition1'].replace('Feedr', 2)
df['Condition1'] = df['Condition1'].replace('Norm', 3)
df['Condition1'] = df['Condition1'].replace('RRNn', 4)
df['Condition1'] = df['Condition1'].replace('RRAn', 5)
df['Condition1'] = df['Condition1'].replace('PosN', 6)
df['Condition1'] = df['Condition1'].replace('PosA', 7)
df['Condition1'] = df['Condition1'].replace('RRNe', 8)
df['Condition1'] = df['Condition1'].replace('RRAe', 9)

#Replace non numeric with numeric in Condition2
df['Condition2'] = df['Condition2'].replace('Artery', 1)
df['Condition2'] = df['Condition2'].replace('Feedr', 2)
df['Condition2'] = df['Condition2'].replace('Norm', 3)
df['Condition2'] = df['Condition2'].replace('RRNn', 4)
df['Condition2'] = df['Condition2'].replace('RRAn', 5)
df['Condition2'] = df['Condition2'].replace('PosN', 6)
df['Condition2'] = df['Condition2'].replace('PosA', 7)
df['Condition2'] = df['Condition2'].replace('RRNe', 8)
df['Condition2'] = df['Condition2'].replace('RRAe', 9)

#Replace non numeric with numeric in BldgType
df['BldgType'] = df['BldgType'].replace('1Fam', 1)
df['BldgType'] = df['BldgType'].replace('2fmCon', 2)
df['BldgType'] = df['BldgType'].replace('Duplex', 3)
df['BldgType'] = df['BldgType'].replace('TwnhsE', 4)
df['BldgType'] = df['BldgType'].replace('Twnhs', 5)

#Replace non numeric with numeric in HouseStyle
df['HouseStyle'] = df['HouseStyle'].replace('1Story', 1)
df['HouseStyle'] = df['HouseStyle'].replace('1.5Fin', 2)
df['HouseStyle'] = df['HouseStyle'].replace('1.5Unf', 3)
df['HouseStyle'] = df['HouseStyle'].replace('2Story', 4)
df['HouseStyle'] = df['HouseStyle'].replace('2.5Fin', 5)
df['HouseStyle'] = df['HouseStyle'].replace('2.5Unf', 6)
df['HouseStyle'] = df['HouseStyle'].replace('SFoyer', 7)
df['HouseStyle'] = df['HouseStyle'].replace('SLvl', 8)

#Replace non numeric with numeric in RoofStyle
df['RoofStyle'] = df['RoofStyle'].replace('Flat', 1)
df['RoofStyle'] = df['RoofStyle'].replace('Gable', 2)
df['RoofStyle'] = df['RoofStyle'].replace('Gambrel', 3)
df['RoofStyle'] = df['RoofStyle'].replace('Hip', 4)
df['RoofStyle'] = df['RoofStyle'].replace('Mansard', 5)
df['RoofStyle'] = df['RoofStyle'].replace('Shed', 6)

#Replace non numeric with numeric in RoofMatl
df['RoofMatl'] = df['RoofMatl'].replace('ClyTile', 1)
df['RoofMatl'] = df['RoofMatl'].replace('CompShg', 2)
df['RoofMatl'] = df['RoofMatl'].replace('Membran', 3)
df['RoofMatl'] = df['RoofMatl'].replace('Metal', 4)
df['RoofMatl'] = df['RoofMatl'].replace('Roll', 5)
df['RoofMatl'] = df['RoofMatl'].replace('Tar&Grv', 6)
df['RoofMatl'] = df['RoofMatl'].replace('WdShake', 7)
df['RoofMatl'] = df['RoofMatl'].replace('WdShngl', 8)

#Replace non numeric with numeric in Exterior1st
df['Exterior1st'] = df['Exterior1st'].replace('AsbShng', 1)
df['Exterior1st'] = df['Exterior1st'].replace('AsphShn', 2)
df['Exterior1st'] = df['Exterior1st'].replace('BrkComm', 3)
df['Exterior1st'] = df['Exterior1st'].replace('BrkFace', 4)
df['Exterior1st'] = df['Exterior1st'].replace('CBlock', 5)
df['Exterior1st'] = df['Exterior1st'].replace('CemntBd', 6)
df['Exterior1st'] = df['Exterior1st'].replace('HdBoard', 7)
df['Exterior1st'] = df['Exterior1st'].replace('ImStucc', 8)
df['Exterior1st'] = df['Exterior1st'].replace('MetalSd', 9)
df['Exterior1st'] = df['Exterior1st'].replace('Other', 10)
df['Exterior1st'] = df['Exterior1st'].replace('Plywood', 11)
df['Exterior1st'] = df['Exterior1st'].replace('PreCast', 12)
df['Exterior1st'] = df['Exterior1st'].replace('Stone', 13)
df['Exterior1st'] = df['Exterior1st'].replace('Stucco', 14)
df['Exterior1st'] = df['Exterior1st'].replace('VinylSd', 16)
df['Exterior1st'] = df['Exterior1st'].replace('Wd Sdng', 17)
df['Exterior1st'] = df['Exterior1st'].replace('Wd Shng', 18)
df['Exterior1st'] = df['Exterior1st'].replace('WdShing', 19)


#Replace non numeric with numeric in Exterior2st
df['Exterior2nd'] = df['Exterior2nd'].replace('AsbShng', 1)
df['Exterior2nd'] = df['Exterior2nd'].replace('AsphShn', 2)
df['Exterior2nd'] = df['Exterior2nd'].replace('BrkComm', 3)
df['Exterior2nd'] = df['Exterior2nd'].replace('BrkFace', 4)
df['Exterior2nd'] = df['Exterior2nd'].replace('CBlock', 5)
df['Exterior2nd'] = df['Exterior2nd'].replace('CmentBd', 6)
df['Exterior2nd'] = df['Exterior2nd'].replace('HdBoard', 7)
df['Exterior2nd'] = df['Exterior2nd'].replace('ImStucc', 8)
df['Exterior2nd'] = df['Exterior2nd'].replace('MetalSd', 9)
df['Exterior2nd'] = df['Exterior2nd'].replace('Other', 10)
df['Exterior2nd'] = df['Exterior2nd'].replace('Plywood', 11)
df['Exterior2nd'] = df['Exterior2nd'].replace('PreCast', 12)
df['Exterior2nd'] = df['Exterior2nd'].replace('Stone', 13)
df['Exterior2nd'] = df['Exterior2nd'].replace('Stucco', 14)
df['Exterior2nd'] = df['Exterior2nd'].replace('VinylSd', 16)
df['Exterior2nd'] = df['Exterior2nd'].replace('Wd Sdng', 17)
df['Exterior2nd'] = df['Exterior2nd'].replace('Wd Shng', 18)
df['Exterior1st'] = df['Exterior1st'].replace('WdShing', 19)
df['Exterior2nd'] = df['Exterior2nd'].replace('Brk Cmn', 20)

#Replace non numeric with numeric in MasVnrType
df['MasVnrType'] = df['MasVnrType'].replace('BrkCmn', 1)
df['MasVnrType'] = df['MasVnrType'].replace('BrkFace', 2)
df['MasVnrType'] = df['MasVnrType'].replace('CBlock', 3)
df['MasVnrType'] = df['MasVnrType'].replace('None', 4)
df['MasVnrType'] = df['MasVnrType'].replace('Stone', 5)

#Replace non numeric with numeric in ExterQual
df['ExterQual'] = df['ExterQual'].replace('Ex', 1)
df['ExterQual'] = df['ExterQual'].replace('Gd', 2)
df['ExterQual'] = df['ExterQual'].replace('TA', 3)
df['ExterQual'] = df['ExterQual'].replace('Fa', 4)
df['ExterQual'] = df['ExterQual'].replace('Po', 5)

#Replace non numeric with numeric in ExterCond
df['ExterCond'] = df['ExterCond'].replace('Ex', 1)
df['ExterCond'] = df['ExterCond'].replace('Gd', 2)
df['ExterCond'] = df['ExterCond'].replace('TA', 3)
df['ExterCond'] = df['ExterCond'].replace('Fa', 4)
df['ExterCond'] = df['ExterCond'].replace('Po', 5)

#Replace non numeric with numeric in Foundation
df['Foundation'] = df['Foundation'].replace('BrkTil', 1)
df['Foundation'] = df['Foundation'].replace('CBlock', 2)
df['Foundation'] = df['Foundation'].replace('PConc', 3)
df['Foundation'] = df['Foundation'].replace('Slab', 4)
df['Foundation'] = df['Foundation'].replace('Stone', 5)
df['Foundation'] = df['Foundation'].replace('Wood', 6)

#Replace non numeric with numeric in BsmtQual
df['BsmtQual'] = df['BsmtQual'].replace('Ex', 1)
df['BsmtQual'] = df['BsmtQual'].replace('Gd', 2)
df['BsmtQual'] = df['BsmtQual'].replace('TA', 3)
df['BsmtQual'] = df['BsmtQual'].replace('Fa', 4)
df['BsmtQual'] = df['BsmtQual'].replace('Po', 5)
df['BsmtQual'] = df['BsmtQual'].replace('NA', 6)
df['BsmtQual'] = df['BsmtQual'].fillna(0)

#Replace non numeric with numeric in BsmtCond
df['BsmtCond'] = df['BsmtCond'].replace('Ex', 1)
df['BsmtCond'] = df['BsmtCond'].replace('Gd', 2)
df['BsmtCond'] = df['BsmtCond'].replace('TA', 3)
df['BsmtCond'] = df['BsmtCond'].replace('Fa', 4)
df['BsmtCond'] = df['BsmtCond'].replace('Po', 5)
df['BsmtCond'] = df['BsmtCond'].replace('NA', 6)
df['BsmtCond'] = df['BsmtCond'].fillna(0)

#Replace non numeric with numeric in BsmtExposure
df['BsmtExposure'] = df['BsmtExposure'].replace('Gd', 1)
df['BsmtExposure'] = df['BsmtExposure'].replace('Av', 2)
df['BsmtExposure'] = df['BsmtExposure'].replace('Mn', 3)
df['BsmtExposure'] = df['BsmtExposure'].replace('No', 4)
df['BsmtExposure'] = df['BsmtExposure'].replace('NA', 5)
df['BsmtExposure'] = df['BsmtExposure'].fillna(0)

#Replace non numeric with numeric in BsmtFinType1
df['BsmtFinType1'] = df['BsmtFinType1'].replace('GLQ', 1)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('ALQ', 2)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('BLQ', 3)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('Rec', 4)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('LwQ', 5)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('Unf', 6)
df['BsmtFinType1'] = df['BsmtFinType1'].replace('NA', 7)
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(0)

#Replace non numeric with numeric in BsmtFinType2
df['BsmtFinType2'] = df['BsmtFinType2'].replace('GLQ', 1)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('ALQ', 2)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('BLQ', 3)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('Rec', 4)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('LwQ', 5)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('Unf', 6)
df['BsmtFinType2'] = df['BsmtFinType2'].replace('NA', 7)
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(0)

#Replace non numeric with numeric in Heating
df['Heating'] = df['Heating'].replace('Floor', 1)
df['Heating'] = df['Heating'].replace('GasA', 2)
df['Heating'] = df['Heating'].replace('GasW', 3)
df['Heating'] = df['Heating'].replace('Grav', 4)
df['Heating'] = df['Heating'].replace('OthW', 5)
df['Heating'] = df['Heating'].replace('Wall', 6)

#Replace non numeric with numeric in HeatingQC
df['HeatingQC'] = df['HeatingQC'].replace('Ex', 1)
df['HeatingQC'] = df['HeatingQC'].replace('Gd', 2)
df['HeatingQC'] = df['HeatingQC'].replace('TA', 3)
df['HeatingQC'] = df['HeatingQC'].replace('Fa', 4)
df['HeatingQC'] = df['HeatingQC'].replace('Po', 5)

#Replace non numeric with numeric in CentralAir
df['CentralAir'] = df['CentralAir'].replace('N', 0)
df['CentralAir'] = df['CentralAir'].replace('Y', 1)

#Replace non numeric with numeric in Electrical
df['Electrical'] = df['Electrical'].replace('SBrkr', 1)
df['Electrical'] = df['Electrical'].replace('FuseA', 2)
df['Electrical'] = df['Electrical'].replace('FuseB', 3)
df['Electrical'] = df['Electrical'].replace('FuseP', 4)
df['Electrical'] = df['Electrical'].replace('Mix', 5)
df['Electrical'] = df['Electrical'].replace('FuseF', 6)


#Replace non numeric with numeric in KitchenQual
df['KitchenQual'] = df['KitchenQual'].replace('Ex', 1)
df['KitchenQual'] = df['KitchenQual'].replace('Gd', 2)
df['KitchenQual'] = df['KitchenQual'].replace('TA', 3)
df['KitchenQual'] = df['KitchenQual'].replace('Fa', 4)
df['KitchenQual'] = df['KitchenQual'].replace('Po', 5)


#Replace non numeric with numeric in FireplaceQu
df['FireplaceQu'] = df['FireplaceQu'].replace('Ex', 1)
df['FireplaceQu'] = df['FireplaceQu'].replace('Gd', 2)
df['FireplaceQu'] = df['FireplaceQu'].replace('TA', 3)
df['FireplaceQu'] = df['FireplaceQu'].replace('Fa', 4)
df['FireplaceQu'] = df['FireplaceQu'].replace('Po', 5)
df['FireplaceQu'] = df['FireplaceQu'].replace('NA', 6)
df['FireplaceQu'] = df['FireplaceQu'].fillna(0)

#Replace non numeric with numeric in FireplaceQu
df['Functional'] = df['Functional'].replace('Typ', 1)
df['Functional'] = df['Functional'].replace('Min1', 2)
df['Functional'] = df['Functional'].replace('Min2', 3)
df['Functional'] = df['Functional'].replace('Mod', 4)
df['Functional'] = df['Functional'].replace('Maj1', 5)
df['Functional'] = df['Functional'].replace('Maj2', 6)
df['Functional'] = df['Functional'].replace('Sev', 7)
df['Functional'] = df['Functional'].replace('Sal', 8)

#Replace non numeric with numeric in GarageType
df['GarageType'] = df['GarageType'].replace('2Types', 1)
df['GarageType'] = df['GarageType'].replace('Attchd', 2)
df['GarageType'] = df['GarageType'].replace('Basment', 3)
df['GarageType'] = df['GarageType'].replace('BuiltIn', 4)
df['GarageType'] = df['GarageType'].replace('CarPort', 5)
df['GarageType'] = df['GarageType'].replace('Detchd', 6)
df['GarageType'] = df['GarageType'].replace('NA', 7)
df['GarageType'] = df['GarageType'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

#Replace non numeric with numeric in GarageFinish
df['GarageFinish'] = df['GarageFinish'].replace('Fin', 1)
df['GarageFinish'] = df['GarageFinish'].replace('RFn', 2)
df['GarageFinish'] = df['GarageFinish'].replace('Unf', 3)
df['GarageFinish'] = df['GarageFinish'].replace('NA', 4)
df['GarageFinish'] = df['GarageFinish'].fillna(0)

#Replace non numeric with numeric in GarageQual
df['GarageQual'] = df['GarageQual'].replace('Ex', 1)
df['GarageQual'] = df['GarageQual'].replace('Gd', 2)
df['GarageQual'] = df['GarageQual'].replace('TA', 3)
df['GarageQual'] = df['GarageQual'].replace('Fa', 4)
df['GarageQual'] = df['GarageQual'].replace('Po', 5)
df['GarageQual'] = df['GarageQual'].replace('NA', 6)
df['GarageQual'] = df['GarageQual'].fillna(0)

#Replace non numeric with numeric in GarageQual
df['GarageCond'] = df['GarageCond'].replace('Ex', 1)
df['GarageCond'] = df['GarageCond'].replace('Gd', 2)
df['GarageCond'] = df['GarageCond'].replace('TA', 3)
df['GarageCond'] = df['GarageCond'].replace('Fa', 4)
df['GarageCond'] = df['GarageCond'].replace('Po', 5)
df['GarageCond'] = df['GarageCond'].replace('NA', 6)
df['GarageCond'] = df['GarageCond'].fillna(0)

#Replace non numeric with numeric in PavedDrive
df['PavedDrive'] = df['PavedDrive'].replace('N', 1)
df['PavedDrive'] = df['PavedDrive'].replace('Y', 2)
df['PavedDrive'] = df['PavedDrive'].replace('P', 3)

#Replace non numeric with numeric in PoolQC
df['PoolQC'] = df['PoolQC'].replace('Ex', 1)
df['PoolQC'] = df['PoolQC'].replace('Gd', 2)
df['PoolQC'] = df['PoolQC'].replace('TA', 3)
df['PoolQC'] = df['PoolQC'].replace('Fa', 4)
df['PoolQC'] = df['PoolQC'].replace('NA', 5)
df['PoolQC'] = df['PoolQC'].fillna(0)

#Replace non numeric with numeric in Fence
df['Fence'] = df['Fence'].replace('GdPrv', 1)
df['Fence'] = df['Fence'].replace('MnPrv', 2)
df['Fence'] = df['Fence'].replace('GdWo', 3)
df['Fence'] = df['Fence'].replace('MnWw', 4)
df['Fence'] = df['Fence'].replace('NA', 5)
df['Fence'] = df['Fence'].fillna(0)


#Replace non numeric with numeric in MiscFeature
df['MiscFeature'] = df['MiscFeature'].replace('Elev', 1)
df['MiscFeature'] = df['MiscFeature'].replace('Gar2', 2)
df['MiscFeature'] = df['MiscFeature'].replace('Othr', 3)
df['MiscFeature'] = df['MiscFeature'].replace('Shed', 4)
df['MiscFeature'] = df['MiscFeature'].replace('TenC', 5)
df['MiscFeature'] = df['MiscFeature'].replace('NA', 6)
df['MiscFeature'] = df['MiscFeature'].fillna(6)

#Replace non numeric with numeric in MiscFeature
df['SaleType'] = df['SaleType'].replace('WD', 1)
df['SaleType'] = df['SaleType'].replace('CWD', 2)
df['SaleType'] = df['SaleType'].replace('VWD', 3)
df['SaleType'] = df['SaleType'].replace('New', 4)
df['SaleType'] = df['SaleType'].replace('COD', 5)
df['SaleType'] = df['SaleType'].replace('Con', 6)
df['SaleType'] = df['SaleType'].replace('ConLw', 7)
df['SaleType'] = df['SaleType'].replace('ConLI', 8)
df['SaleType'] = df['SaleType'].replace('ConLD', 9)
df['SaleType'] = df['SaleType'].replace('Oth', 10)

#Replace non numeric with numeric in SaleCondition
df['SaleCondition'] = df['SaleCondition'].replace('Normal', 1)
df['SaleCondition'] = df['SaleCondition'].replace('Abnorml', 2)
df['SaleCondition'] = df['SaleCondition'].replace('AdjLand', 3)
df['SaleCondition'] = df['SaleCondition'].replace('Alloca', 4)
df['SaleCondition'] = df['SaleCondition'].replace('Family', 5)
df['SaleCondition'] = df['SaleCondition'].replace('Partial', 6)

df = df.fillna(0)

df.to_csv("newTest.csv")