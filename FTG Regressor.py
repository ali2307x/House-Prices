import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix


class PreProc(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame, y=None):
            # Fix data issues and inconsistencies
            X['MSZoning'] = X['MSZoning'].replace('C (all)', 'C')
            X['Exterior2nd'] = X['Exterior2nd'].replace('Wd Shng', 'WdShing')
            X['Exterior2nd'] = X['Exterior2nd'].replace('Brk Cmn', 'BrkComm')
            X['Exterior2nd'] = X['Exterior2nd'].replace('CmentBd', 'CemntBd')
            X.drop('Id', axis=1, inplace=True)
            # Circumvent Issues with NA
            X['MSZoning'] = X['MSZoning'].fillna('RL')
            X['LotFrontage'] = X['LotFrontage'].fillna(0)
            X['Alley'] = X['Alley'].fillna('NAX')
            X['MasVnrType'] = X['MasVnrType'].fillna('NAX')
            X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
            X['BsmtQual'] = X['BsmtQual'].fillna('NAX')
            X['BsmtCond'] = X['BsmtCond'].fillna('NAX')
            X['BsmtExposure'] = X['BsmtExposure'].fillna('NAX')
            X['BsmtFinType1'] = X['BsmtFinType1'].fillna('NAX')
            X['BsmtFinType2'] = X['BsmtFinType2'].fillna('NAX')
            X['Electrical'] = X['Electrical'].fillna('NAX')
            X['FireplaceQu'] = X['FireplaceQu'].fillna('NAX')
            X['GarageType'] = X['GarageType'].fillna('NAX')
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
            X['GarageFinish'] = X['GarageFinish'].fillna('NAX')
            X['GarageQual'] = X['GarageQual'].fillna('NAX')
            X['GarageCond'] = X['GarageCond'].fillna('NAX')
            X['PoolQC'] = X['PoolQC'].fillna('NAX')
            X['Fence'] = X['Fence'].fillna('NAX')
            X['MiscFeature'] = X['MiscFeature'].fillna('NAX')
            X['SaleType'] = X['SaleType'].fillna('WD')
            X['Utilities'] = X['Utilities'].fillna('AllPub')
            X['KitchenQual'] = X['KitchenQual'].fillna('TA')
            X['Functional'] = X['Functional'].fillna('Typ')

            # New Features
            X['Age'] = X['YrSold'] - X['YearBuilt']
            X['WasRemod'] = X['YearRemodAdd'] > X['YearBuilt']
            X['SinceRemod'] = X['YrSold'] - X['YearRemodAdd']
            X['2Cond'] = pd.DataFrame([X['Condition1'] != X['Condition2'], X['Condition1'] != 'Norm', X['Condition2'] != 'Norm']).T.all(axis='columns')
            X['2Ext'] = X['Exterior1st'] != X['Exterior2nd']
            X['AnotherBsmt'] = X['BsmtFinType2'] != 'Unf'
            X['TotBsmtFn'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
            X['BsmtBathTot'] = X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath']
            X['TotBath'] = X['FullBath'] + 0.5 * X['HalfBath']
            # X['CarGarageArea'] = X['GarageArea'] / X['GarageCars']

            return X


class NomCatTransformer(BaseEstimator, TransformerMixin):

        cat_binarizers = {}

        def fit(self, X, y=None):
            for col in X.columns:
                label_binarizer = LabelBinarizer()
                self.cat_binarizers[col] = label_binarizer.fit(X[col])
            return self

        def transform(self, X, y=None):
            cat_features = []
            for col in X.columns:
                trans_data = self.cat_binarizers[col].transform(X[col])
                if len(cat_features) == 0:
                    cat_features = trans_data
                else:
                    cat_features = np.c_[cat_features, trans_data]
            return cat_features


class OrdCatTransformer(BaseEstimator, TransformerMixin):

        norm_scaler = Normalizer()
        mode = 'train'

        def __init__(self, mode='train'):
            self.mode = mode

        @staticmethod
        def quality_encoder(X, col):

            X[col] = X[col].replace('NAX', -1)
            # X[col] = X[col].replace('No', 0)
            X[col] = X[col].replace('Po', 1)
            X[col] = X[col].replace('Fa', 2)
            X[col] = X[col].replace('TA', 3)
            X[col] = X[col].replace('Gd', 4)
            X[col] = X[col].replace('Ex', 5)

            return X[col]

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X['ExterQual'] = self.quality_encoder(X, 'ExterQual')
            X['ExterCond'] = self.quality_encoder(X, 'ExterCond')
            X['BsmtQual'] = self.quality_encoder(X, 'BsmtQual')
            X['BsmtCond'] = self.quality_encoder(X, 'BsmtCond')
            X['HeatingQC'] = self.quality_encoder(X, 'HeatingQC')
            X['KitchenQual'] = self.quality_encoder(X, 'KitchenQual')
            X['FireplaceQu'] = self.quality_encoder(X, 'FireplaceQu')
            X['GarageQual'] = self.quality_encoder(X, 'GarageQual')
            X['GarageCond'] = self.quality_encoder(X, 'GarageCond')
            X['PoolQC'] = self.quality_encoder(X, 'PoolQC')

            X['PavedDrive'] = X['PavedDrive'].replace('N', 0)
            X['PavedDrive'] = X['PavedDrive'].replace('P', 1)
            X['PavedDrive'] = X['PavedDrive'].replace('Y', 2)

            X['Functional'] = X['Functional'].replace('Sal', 0)
            X['Functional'] = X['Functional'].replace('Sev', 1)
            X['Functional'] = X['Functional'].replace('Maj2', 2)
            X['Functional'] = X['Functional'].replace('Maj1', 3)
            X['Functional'] = X['Functional'].replace('Mod', 4)
            X['Functional'] = X['Functional'].replace('Min2', 5)
            X['Functional'] = X['Functional'].replace('Min1', 6)
            X['Functional'] = X['Functional'].replace('Typ', 7)

            X['GarageFinish'] = X['GarageFinish'].replace('No', 0)
            X['GarageFinish'] = X['GarageFinish'].replace('Unf', 1)
            X['GarageFinish'] = X['GarageFinish'].replace('RFn', 2)
            X['GarageFinish'] = X['GarageFinish'].replace('Fin', 3)

            X['BsmtExposure'] = X['BsmtExposure'].replace('NAX', 0)
            X['BsmtExposure'] = X['BsmtExposure'].replace('No', 1)
            X['BsmtExposure'] = X['BsmtExposure'].replace('Mn', 2)
            X['BsmtExposure'] = X['BsmtExposure'].replace('Av', 3)
            X['BsmtExposure'] = X['BsmtExposure'].replace('Gd', 4)

            X['BsmtFinType1'] = X['BsmtFinType1'].replace('NAX', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('Unf', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('LwQ', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('Rec', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('BLQ', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('ALQ', 0)
            X['BsmtFinType1'] = X['BsmtFinType1'].replace('GLQ', 0)

            X['BsmtFinType2'] = X['BsmtFinType2'].replace('NAX', 0)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('Unf', 1)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('LwQ', 2)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('Rec', 3)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('BLQ', 4)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('ALQ', 5)
            X['BsmtFinType2'] = X['BsmtFinType2'].replace('GLQ', 6)

            X['GarageFinish'] = X['GarageFinish'].replace('NAX', 0)
            X['GarageFinish'] = X['GarageFinish'].replace('Unf', 1)
            X['GarageFinish'] = X['GarageFinish'].replace('RFn', 2)
            X['GarageFinish'] = X['GarageFinish'].replace('Fin', 3)

            if self.mode == 'train':
                return self.norm_scaler.fit_transform(X)
            else:
                return self.norm_scaler.transform(X)


def house_class_picker(price: float):
    if price < 90000:
        return 'cheap'
    elif price < 230000:
        return 'medium'
    elif price < 300000:
        return 'expensive'
    else:
        return 'very expensive'


data = pd.read_csv('Data/train.csv',  delimiter=',', quotechar='"', skipinitialspace=True)

data: pd.DataFrame = PreProc().fit_transform(data)

price_brackets = (data['SalePrice']/33000).astype(int)

data['PriceClass'] = data['SalePrice'].copy()
data['PriceClass'] = data['PriceClass'].apply(house_class_picker)

price_brackets.where(price_brackets < 15, 15, inplace=True)

nom_cat_features = data.loc[:, ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                                'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'Fence', 'MiscFeature',
                                'MoSold', 'SaleType', 'SaleCondition', 'WasRemod', 'AnotherBsmt', '2Cond', '2Ext']]

ord_cat_features = data.loc[:, ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual',
                                'GarageCond', 'PavedDrive', 'PoolQC']]

scaled_num_features = data.loc[:, ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                                   'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                                   'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'Age', 'SinceRemod',
                                   'SinceRemod', 'TotBsmtFn', 'BsmtBathTot', 'TotBath', 'YearBuilt', 'YrSold']].replace('No', 0)

minmax_num_features = data.loc[:, ['OverallQual', 'OverallCond']].replace('No', 0)

num_scaler = StandardScaler()
num_norm = Normalizer()
minmax_scaler = MinMaxScaler()
nom_cat_trans = NomCatTransformer()
ord_cat_trans = OrdCatTransformer(mode='train')


scaled_num_features = num_norm.fit_transform(scaled_num_features.to_numpy())
minmax_num_features = minmax_scaler.fit_transform(minmax_num_features.to_numpy())
nom_cat_features = nom_cat_trans.fit_transform(nom_cat_features)
ord_cat_features = ord_cat_trans.fit_transform(ord_cat_features)

X = np.c_[scaled_num_features, minmax_num_features, nom_cat_features, ord_cat_features]
y = data['PriceClass'].to_numpy()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(iter(split.split(data, price_brackets)))
X_train = X[train_index]
y_train = y[train_index]
X_test = X[test_index]
y_test = y[test_index]


######## Price Class Classifier ##########


rnd_for_clf = RandomForestClassifier(max_depth=10, max_features=1.0, n_jobs=-1, criterion="entropy", n_estimators=300)

rnd_for_clf.fit(X_train, y_train)
preds = rnd_for_clf.predict(X_test)

# print(np.round(accuracy_score(y_test, preds)*100, decimals=2))
# print(confusion_matrix(y_test, preds, labels=['cheap', 'medium', 'expensive', 'very expensive']))


##########################################



######## Price Value Regressor ##########

regressors = {}
price_class = data['PriceClass'].to_numpy()
price_class_train = price_class[train_index]
price_class_test = price_class[test_index]
y = data['SalePrice'].to_numpy()
y_train = y[train_index]
y_test = y[test_index]


for class_name in data['PriceClass'].unique():
    regressors[class_name] = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                                         learning_rate=0.25, loss='ls', max_depth=5, max_features=0.8,
                                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                         min_impurity_split=None, min_samples_leaf=1,
                                                         min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                         n_estimators=200, presort='auto', random_state=None,
                                                         subsample=0.9, verbose=0, warm_start=False)

    selected_values = price_class_train == class_name
    temp_X = X_train[selected_values]
    temp_y = y_train[selected_values]

    regressors[class_name].fit(temp_X, temp_y)

main_regressor = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                                         learning_rate=0.25, loss='ls', max_depth=5, max_features=0.8,
                                                         max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                         min_impurity_split=None, min_samples_leaf=1,
                                                         min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                         n_estimators=200, presort='auto', random_state=None,
                                                         subsample=0.9, verbose=0, warm_start=False)

main_regressor.fit(X_train, y_train)

##########################################


preds = []

pred_class_names = rnd_for_clf.predict(X_test)

for i in range(len(X_test)):
    class_name = pred_class_names[i]

    predicted_price = 0
    predicted_price += 0.2 * regressors[class_name].predict([X_test[i]])[0]
    predicted_price += 0.8 * main_regressor.predict([X_test[i]])[0]

    preds.append(predicted_price)

# preds = np.reshape(preds, (-1, 1))

print(np.sqrt(mean_squared_error(y_test, preds)).round(2))


"""

preds = grad_reg.predict(X_test)


diffs = preds - y_test
percs = diffs / y_test

# sb.distplot(diffs, kde=False)
# sb.distplot(percs)
# plt.show()

f = open("benchmark.csv", "w")
f.write('Pred,Actual,Diff,Perc\n')
for pred, actual, diff, perc in zip(preds, y_test, diffs, percs):
    f.write(str(pred) + ',' + str(actual) + ',' + str(diff) + ',' + str(perc) + '\n')

f.close()






########## TEST ###########



data = pd.read_csv('Data/test.csv',  delimiter=',', quotechar='"', skipinitialspace=True)

ids = data['Id']

data: pd.DataFrame = PreProc().transform(data)

nom_cat_features = data.loc[:, ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                                'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'Fence', 'MiscFeature',
                                'MoSold', 'SaleType', 'SaleCondition', 'WasRemod', 'AnotherBsmt', '2Cond', '2Ext']]


ord_cat_features = data.loc[:, ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual',
                                'GarageCond', 'PavedDrive', 'PoolQC']]

scaled_num_features = data.loc[:, ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                                   'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                                   'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'Age', 'SinceRemod',
                                   'SinceRemod', 'TotBsmtFn', 'BsmtBathTot', 'TotBath', 'YearBuilt', 'YrSold']].replace('No', 0).fillna(0)

minmax_num_features = data.loc[:, ['OverallQual', 'OverallCond']].replace('No', 0)

ord_cat_trans.mode = 'test'

scaled_num_features = num_norm.transform(scaled_num_features.to_numpy())
minmax_num_features = minmax_scaler.transform(minmax_num_features.to_numpy())
nom_cat_features = nom_cat_trans.transform(nom_cat_features)
ord_cat_features = ord_cat_trans.transform(ord_cat_features)

X = np.c_[scaled_num_features, minmax_num_features, nom_cat_features, ord_cat_features]

print(X.shape)
preds = grad_reg.predict(X)

f = open("results.csv", "w")
f.write('Id,SalePrice\n')
for id, price in zip(ids, preds):
    f.write(str(id) + ',' + str(price) + '\n')

f.close()
"""