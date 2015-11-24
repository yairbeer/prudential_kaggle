from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
import xgboostlib.xgboost as xgboost
from sklearn.decomposition import PCA

__author__ = 'YBeer'

train_result = pd.DataFrame.from_csv("train_result.csv")
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
# train_result = pd.get_dummies(train_result)
train_result = np.array(train_result).ravel()
train_result_xgb = np.zeros(train_result.shape)
for i in range(1, len(result_ind)):
    train_result_xgb += (train_result == result_ind[i]) * i
# print train_result_xgb

train = pd.DataFrame.from_csv("train_dummied_150_sep_dep_fln_b_r_v5.csv").astype('float')
train.fillna(0)
train_arr = np.array(train)
col_list = list(train.columns.values)

test = pd.DataFrame.from_csv("test_dummied_150_sep_dep_fln_b_r_v5.csv").astype('float')
test.fillna(0)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

print 'absing'
for i in range(train_arr.shape[0]):
    for j in range(train_arr.shape[1]):
        train_arr[i, j] = np.abs(train_arr[i, j])

chi2_params = chi2(train_arr, train_result)

for i in range(train_arr.shape[1]):
    print col_list[i], chi2_params[0][i]

del train_arr

best_metric = 10
best_params = []
param_grid = {'silent': [1], 'nthread': [4], 'num_class': [38], 'eval_metric': ['mlogloss'], 'eta': [0.1],
              'objective': ['multi:softprob'], 'max_depth': [5], 'chi2_lim': [0], 'num_round': [400],
              'subsample': [0.7]}

for params in ParameterGrid(param_grid):
    print params

    # filtering low chi2 cols
    chi2_lim = params['chi2_lim']
    chi2_cols = []
    for i in range(train.shape[1]):
        if chi2_params[0][i] > chi2_lim:
            chi2_cols.append(col_list[i])

    print len(chi2_cols), ' chi2 columns'
    train = train[chi2_cols]
    train = np.array(train)
    test = test[chi2_cols]
    test = np.array(test)

    # Standardizing
    stding = StandardScaler()
    train = stding.fit_transform(train)
    test = stding.transform(test)

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
    predicted_results = predicted_results.reshape(test.shape[0], 38)

    print 'writing to file'
    submission_file = pd.DataFrame.from_csv("sample_submission.csv")
    submission_cols = list(submission_file.columns.values)
    submission_vals = map(lambda x: int(x.split("_")[1]), submission_cols)

    submission_table = np.zeros(predicted_results.shape)
    for i in range(predicted_results.shape[1]):
        for j in range(predicted_results.shape[1]):
            if submission_vals[i] == result_ind[j]:
                print 'adding triptype ', submission_vals[i]
                submission_table[:, i] = predicted_results[:, j]

    submission_file[list(submission_file.columns.values)] = submission_table
    submission_file.to_csv("chi2_feature_select_xgboost_5depth_v5_ss07.csv")
