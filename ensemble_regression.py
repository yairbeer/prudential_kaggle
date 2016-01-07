from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
import xgboostlib.xgboost as xgboost
import glob
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.cross_validation import StratifiedKFold
import scipy.optimize as optimize

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


def ranking(predictions, split_index):
    # print predictions
    ranked_predictions = np.ones(predictions.shape)

    for i in range(1, len(split_index)):
        cond = (split_index[i-1] <= predictions) * 1 * (predictions < split_index[i])
        ranked_predictions[cond.astype('bool')] = i+1
    cond = (predictions >= split_index[-1])
    ranked_predictions[cond] = len(split_index) + 1
    # print cond
    # print ranked_predictions
    return ranked_predictions


train_result = pd.DataFrame.from_csv("train_result.csv")
# print train_result['Response'].value_counts()

col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel()

# combining meta_estimators
train = glob.glob('meta_train*')
print train
for i in range(len(train)):
    train[i] = pd.DataFrame.from_csv(train[i])
train = pd.concat(train, axis=1)
train = np.array(train)

test = glob.glob('meta_test*')
print test
for i in range(len(test)):
    test[i] = pd.DataFrame.from_csv(test[i])
test = pd.concat(test, axis=1)
test = np.array(test)

# print train_result.shape[1], ' categorial'
print train.shape[1], ' columns'

# 4th
# splitter = [2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844, 6.17412558, 6.79373477]
# nelder mead opt
splitter = np.array([2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844, 6.17412558, 6.79373477])
riskless_splitter = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])


def opt_cut_v2_cv(x):
    case = np.array(ranking(predicted_results, x)).astype('int')
    score = -1 * quadratic_weighted_kappa(y_test, case, 1, 8)
    # print score
    return score

best_risk = 0
best_score = 0
best_splitter = 0
risk = 0.95

regressor = LinearRegression(fit_intercept=True)
param_grid = [
              {'risk': [1]}
             ]

print 'start CV'
for params in ParameterGrid(param_grid):
    print params

    # CV
    cv_n = 12
    kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)
    it_splitter = []
    metric = []
    for train_index, test_index in kf:
        X_train, X_test = train[train_index, :], train[test_index, :]
        y_train, y_test = train_result[train_index].ravel(), train_result[test_index].ravel()
        # train machine learning

        regressor.fit(X_train, y_train)

        # predict
        predicted_results = regressor.predict(X_test)
        case = np.array(ranking(predicted_results, [2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844,
                                                    6.17412558, 6.79373477])).astype('int')
        # train machine learning
        res = optimize.minimize(opt_cut_v2_cv, [2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844,
                                                6.17412558, 6.79373477], method='Nelder-Mead',
                                # options={'disp': True}
                                )
        # print res.x
        cur_splitter = list(params['risk'] * res.x + (1 - params['risk']) * riskless_splitter)
        it_splitter.append(cur_splitter)
        # print cur_splitter
        classified_predicted_results = np.array(ranking(predicted_results, cur_splitter)).astype('int')
        # predict
        print quadratic_weighted_kappa(y_test, classified_predicted_results, 1, 8)
        metric.append(quadratic_weighted_kappa(y_test, classified_predicted_results, 1, 8))
    print 'The quadratic weighted kappa is: ', np.mean(metric)
    if np.mean(metric) > best_score:
        print 'new best risk'
        best_score = np.mean(metric)
        best_risk = params['risk']
        it_splitter = np.array(it_splitter)
        best_splitter = np.average(it_splitter, axis=0)

regressor.fit(train, train_result)
print 'The regression coefs are:'
print regressor.coef_, regressor.intercept_
print 'the best risk is: %f' % best_risk
print 'the best quadratic weighted kappa is: %f' % best_score
# predict
predicted_results = regressor.predict(test)

final_splitter = list(splitter * best_risk + riskless_splitter * (1 - best_risk))

print 'writing to file'
classed_results = np.array(ranking(predicted_results, final_splitter)).astype('int')
submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file['Response'] = classed_results

print submission_file['Response'].value_counts()

submission_file.to_csv("ensemble_linear_regression.csv")

# class + reg, dum + not dummy

# regression
# only bossting regressor, no intercept: 0.662372196259
# only bossting regressor, with intercept: 0.662542431915
# only bossting regressor + class, no intercept: 0.66344342224, works bad on test
# only bossting regressor + class, with intercept: 0.663248699375

# added linear regression
# only bossting regressor + linear, with intercept: 0.662427982667

# added nedler mead optimized spliter: 0.666313598761

# added RF depth = 10: 0.663928238356
# added RF depth = 20: 0.664636681173

# added risk

# added RF depth = 30, risk = 1: 0.668113299164

# moved to n_CV = 12
# risk = 1, QWK = 0.667689

# added RF depth = 40, risk = 1: 0.668522

# added best splitter from CV, nelder mead opt splitter was better