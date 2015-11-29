from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboostlib.xgboost as xgboost
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

__author__ = 'YBeer'


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def ranking(predictions, split_index):
    predictions = pd.Series(predictions)
    ranked_predictions = predictions.copy()
    ranked_predictions.iloc[predictions < split_index[0]] = 1
    for i in range(1, (len(split_index) - 1)):
        ranked_predictions.iloc[split_index[i-1] <= predictions < split_index[i]] = i+1
    ranked_predictions.iloc[predictions >= split_index[-1]] = len(split_index-1)
    return ranked_predictions

train_result = pd.DataFrame.from_csv("train_result.csv")
print train_result['Response'].value_counts()

col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel() - 1

train = pd.DataFrame.from_csv("train_dummied_v2.csv").astype('float')
train.fillna(999)
train_arr = np.array(train)
col_list = list(train.columns.values)

test = pd.DataFrame.from_csv("test_dummied_v2.csv").astype('float')
test.fillna(999)
test_arr = np.array(test)


# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# Standardizing
stding = StandardScaler()
train = stding.fit_transform(train_arr)
test = stding.transform(test_arr)

param_grid = [
              {'silent': [1], 'nthread': [3], 'num_class': [8], 'eval_metric': ['mlogloss'], 'eta': [0.03],
               'objective': ['multi:softprob'], 'max_depth': [7], 'num_round': [750],
               'subsample': [0.75]}
             ]

# max_depth = 3, num_round = 1600; max_depth = 5, num_round = 1200; max_depth = 7, num_round = 750;
# max_depth = 9, num_round = 500, max_depth = 11, num_round = 360
best_metric = 10
best_params = []
best_metatrain = None
print 'start CV'
for params in ParameterGrid(param_grid):
    print params
    # CV
    cv_n = 8
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)

    meta_train = np.ones((train_arr.shape[0], 8))
    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train_arr[train_index, :], train_arr[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning
        # train machine learning
        xg_train = xgboost.DMatrix(X_train, label=y_train)
        xg_test = xgboost.DMatrix(X_test, label=y_test)

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]

        num_round = params['num_round']
        xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

        # predict
        predicted_results = xgclassifier.predict(xg_test)
        predicted_results = predicted_results.reshape(X_test.shape[0], 8)
        # print pd.Series(predicted_results).value_counts()
        # print pd.Series(y_test).value_counts()
        # print quadratic_weighted_kappa(y_test, predicted_results)
        metric.append(log_loss(y_test, predicted_results))
        meta_train[test_index, :] = predicted_results

    print 'The log loss is: ', np.mean(metric)
    if np.mean(metric) < best_metric:
        best_metric = np.mean(metric)
        best_params = params
        best_metatrain = meta_train
        print best_metatrain
    print 'The best metric is: ', best_metric, 'for the params: ', best_params

pd.DataFrame(best_metatrain).to_csv('meta_train_boost_classification.csv')
# train machine learning
xg_train = xgboost.DMatrix(train_arr, label=train_result)
xg_test = xgboost.DMatrix(test_arr)

watchlist = [(xg_train, 'train')]

num_round = best_params['num_round']
xgclassifier = xgboost.train(best_params, xg_train, num_round, watchlist);

# predict
predicted_results = xgclassifier.predict(xg_test)
predicted_results = predicted_results.reshape(test_arr.shape[0], 8)
pd.DataFrame(predicted_results).to_csv('meta_test_boost_classification.csv')

# print 'writing to file'
# submission_file = pd.DataFrame.from_csv("sample_submission.csv")
# submission_file['Response'] = predicted_results
#
# print submission_file['Response'].value_counts()
#
# submission_file.to_csv("xgboost_%sdepth_regression.csv" % best_params['max_depth'])
