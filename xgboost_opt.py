from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import hamming_loss
import xgboostlib.xgboost as xgboost

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv('train_result.csv')
# print train_result['Response'].value_counts()
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel()
train_result_xgb = train_result - 1

train = pd.DataFrame.from_csv("train_dummied.csv")
train.fillna(9999)
train_arr = np.array(train)
col_list = list(train.columns.values)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# Standardizing
stding = StandardScaler()
train_arr = stding.fit_transform(train_arr)

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [8], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [3, 5, 7], 'num_round': [300], 'subsample': [0.5, 0.75, 1]}

for params in ParameterGrid(param_grid):
    print params

    print 'start CV'

    # CV
    X_train, X_test, y_train, y_test = train_test_split(train_arr, train_result_xgb, test_size=0.25, random_state=1)
    metric = []

    # train machine learning
    xg_train = xgboost.DMatrix(X_train, label=y_train)
    xg_test = xgboost.DMatrix(X_test, label=y_test)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    class_pred = xgclassifier.predict(xg_test)

    # evaluate
    # print hamming_loss(y_test, class_pred)
    metric = hamming_loss(y_test, class_pred)

    print 'The log loss is: ', metric
    if metric < best_metric:
        best_metric = metric
        best_params = params
    print 'The best metric is:', best_metric, 'for the params:', best_params
