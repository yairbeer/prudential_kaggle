from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
import xgboostlib.xgboost as xgboost

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
print train_result['Response'].value_counts()

col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()
train_result_xgb = train_result - 1
# print train_result_xgb

train = pd.DataFrame.from_csv("train_dummied.csv").astype('float')
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

test = pd.DataFrame.from_csv("test_dummied.csv").astype('float')
test.fillna(9999)
test_arr = np.array(test)


# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train_arr)
test = stding.transform(test_arr)

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [8], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softmax'], 'max_depth': [5], 'num_round': [100],
              'subsample': [0.7]}

for params in ParameterGrid(param_grid):
    print params

    # # PCA decomposition
    # pcaing = PCA()
    # train = pcaing.fit_transform(train)
    # test = pcaing.transform(test)

    # train machine learning
    xg_train = xgboost.DMatrix(train, label=train_result_xgb)
    xg_test = xgboost.DMatrix(test)

    watchlist = [(xg_train, 'train')]

    num_round = params['num_round']
    xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

    # predict
    predicted_results = xgclassifier.predict(xg_test)
    predicted_results += 1
    predicted_results = predicted_results.astype('int')
    print predicted_results

    print 'writing to file'
    submission_file = pd.DataFrame.from_csv("sample_submission.csv")
    submission_file['Response'] = predicted_results

    print submission_file['Response'].value_counts()

    submission_file.to_csv("xgboost_5depth_v5_ss07.csv")
