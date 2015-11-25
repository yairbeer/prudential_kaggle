from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

__author__ = 'YBeer'

train = pd.DataFrame.from_csv("xgboost_train_7_probabilities.csv")
train_arr = np.array(train)
col_list = list(train.columns.values)


train_result = pd.DataFrame.from_csv("train_result.csv")
# print train_result['Response'].value_counts()
train_result = np.array(train_result).ravel()
# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# Standardizing
stding = StandardScaler()
train_arr = stding.fit_transform(train_arr)

best_metric = 10
best_params = []

param_grid = {'n_estimators': [400], 'max_depth': [3, 4, 5, 6, 7], 'max_features': [0.75], 'fit_const': [0.5],
              'learning_rate': [0.03], 'subsample': [1.0, 0.9, 0.8, 0.7]}

for params in ParameterGrid(param_grid):
    regressor = GradientBoostingRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                          max_features=params['max_features'], learning_rate=params['learning_rate'])

    print 'start CV'
    print params
    # CV
    cv_n = 4
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning
        regressor.fit(X_train, y_train)

        # predict
        class_pred = regressor.predict(X_test)
        class_pred += params['fit_const']
        class_pred = np.floor(class_pred)
        # evaluate
        # print mean_squared_error(y_test, class_pred)
        metric.append(mean_squared_error(y_test, class_pred))
        # print np.hstack((y_test.reshape(y_test.shape[0], 1), class_pred.reshape(y_test.shape[0], 1)))
    print 'The RMSE is: ', np.mean(metric)
    if np.mean(metric) < best_metric:
        best_metric = np.mean(metric)
        best_params = params
    print 'The best metric is: ', best_metric, 'for the params: ', best_params

# The best metric is:  3.31904137794 for the params:  {'max_features': 0.75, 'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 4}

print 'training test file'
test = pd.DataFrame.from_csv("xgboost_test_7_probabilities.csv")
test_arr = np.array(test)
test_arr = stding.transform(test_arr)

regressor = GradientBoostingRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                      max_features=best_params['max_features'],
                                      learning_rate=best_params['learning_rate'])
regressor.fit(train_arr, train_result)
reg_pred = regressor.predict(test_arr)

reg_pred += best_params['fit_const']
reg_pred = np.floor(reg_pred).astype('int')

print 'writing to file'
submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file['Response'] = reg_pred

print submission_file['Response'].value_counts()

submission_file.to_csv("xgboost_reg_%sdepth.csv" % params['max_depth'])
