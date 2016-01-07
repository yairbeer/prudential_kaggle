import pandas as pd
import numpy as np
import scipy.optimize as optimize
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
import matplotlib.pyplot as plt

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
# input probabilities
train_probabilities = pd.DataFrame.from_csv("meta_class_train_boost.csv")
train_probabilities_array = np.array(train_probabilities)
avail_results = np.repeat(np.arange(1, 9).reshape(1, 8), train_probabilities_array.shape[0], axis=0)
train_averages = np.sum(train_probabilities_array * avail_results, axis=1)
train_averages = np.repeat(train_averages.reshape(train_probabilities_array.shape[0], 1), 8, axis=1)
train_std = np.sum((train_averages - (avail_results * train_probabilities_array)) ** 2, axis=1)
# print train_std
# plt.hist(train_std)
# plt.show()
test_probabilities = pd.DataFrame.from_csv("meta_class_test_boost.csv")

std_threshhold = 150
below_thresh = train_std < std_threshhold
print np.sum(below_thresh)
above_thresh = train_std >= std_threshhold
print np.sum(above_thresh)

train_result = pd.DataFrame.from_csv("train_result.csv")
# print train_result
# print train_result['Response'].value_counts()
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel()
train_result_blw_thrsh = train_result[below_thresh]
train_result_abv_thrsh = train_result[above_thresh]

base_splitter = np.arange(7) + 1.5

train_prediction = pd.DataFrame.from_csv("meta_train_boost_regression_8_deep.csv")
# print train_prediction
# print pd.concat([train_probabilities, train_prediction], axis=1)
train_prediction = np.array(train_prediction).ravel()

train_prediction_blw_thrsh = train_prediction[below_thresh]
# print zip(train_prediction_blw_thrsh, train_result_blw_thrsh)
train_prediction_abv_thrsh = train_prediction[above_thresh]
basecase_train = ranking(train_prediction, base_splitter)
basecase_train_blw_thrsh = basecase_train[below_thresh]
basecase_train_abv_thrsh = basecase_train[above_thresh]

train_value_count = np.array(pd.Series(basecase_train).value_counts())
train_value_count = train_value_count.astype('float') / np.sum(train_value_count)
test_prediction = pd.DataFrame.from_csv("meta_test_boost_regression_8_deep.csv")
test_prediction = np.array(test_prediction).ravel()
basecase_test = ranking(test_prediction, base_splitter)
test_value_count = np.array(pd.Series(basecase_test).value_counts())
test_value_count = test_value_count.astype('float') / np.sum(test_value_count)
# print train_value_count
# print test_value_count
#
# print quadratic_weighted_kappa(train_result, basecase_train)

x0_range = np.arange(-5, 6, 0.5)
x1_range = np.arange(-1.5, 1.75, 0.25)
x2_range = np.arange(-0.5, 0.6, 0.1)
x3_range = np.arange(-0.1, 0.2, 0.1)
bestcase = 0
bestscore = 0

print 'start cubic optimization for below %f std threshhold' % std_threshhold
# optimize classifier
for x0 in x0_range:
    for x1 in x1_range:
        for x2 in x2_range:
            for x3 in x3_range:
                case = np.array(ranking(train_prediction_blw_thrsh, (x0 + x1 * base_splitter + x2 * base_splitter**2 +
                                                                     x3 * base_splitter**3))).astype('int')
                score = quadratic_weighted_kappa(train_result_blw_thrsh, case)
                if score > bestscore:
                    bestscore = score
                    bestcase = case
                    print 'For splitter ', (x0 + x1 * base_splitter + x2 * base_splitter**2 +
                                                           x3 * base_splitter**3)
                    print 'Variables x0 = %f, x1 = %f, x2 = %f, x3 = %f' % (x0, x1, x2, x3)
                    print 'The score is %f' % bestscore

bestcase = 0
bestscore = 0
print 'start cubic optimization for above %f std threshhold' % std_threshhold
# optimize classifier
for x0 in x0_range:
    for x1 in x1_range:
        for x2 in x2_range:
            for x3 in x3_range:
                case = np.array(ranking(train_prediction_abv_thrsh, (x0 + x1 * base_splitter + x2 * base_splitter**2 +
                                                                     x3 * base_splitter**3))).astype('int')
                score = quadratic_weighted_kappa(train_result_abv_thrsh, case)
                if score > bestscore:
                    bestscore = score
                    bestcase = case
                    print 'For splitter ', (x0 + x1 * base_splitter + x2 * base_splitter**2 +
                                                           x3 * base_splitter**3)
                    print 'Variables x0 = %f, x1 = %f, x2 = %f, x3 = %f' % (x0, x1, x2, x3)
                    print 'The score is %f' % bestscore


def opt_cut_v2(x, *args):
    train_prediction, train_result = args
    case = np.array(ranking(train_prediction, x)).astype('int')
    score = -1 * quadratic_weighted_kappa(train_result, case)
    # print score
    return score

splitter = np.array([2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844, 6.17412558, 6.79373477])
riskless_splitter = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

# res = optimize.minimize(opt_cut_v2, splitter, args=(train_prediction, train_result),
#                         method='Nelder-Mead', options={'disp': True})
# print res.x
res = optimize.minimize(opt_cut_v2, splitter, args=(train_prediction_blw_thrsh, train_result_blw_thrsh),
                        method='Nelder-Mead', options={'disp': True})
print res.x
case_train_blw_thrsh = ranking(train_prediction_blw_thrsh, res.x)
res = optimize.minimize(opt_cut_v2, base_splitter, args=(train_prediction_abv_thrsh, train_result_abv_thrsh),
                        method='Nelder-Mead', options={'disp': True})
print res.x
case_train_abv_thrsh = ranking(train_prediction_abv_thrsh, res.x)

case_train = np.vstack((case_train_blw_thrsh.reshape(case_train_blw_thrsh.shape[0], 1),
                        case_train_abv_thrsh.reshape(case_train_abv_thrsh.shape[0], 1)))
train_result = np.vstack((train_result_blw_thrsh.reshape(train_result_blw_thrsh.shape[0], 1),
                          train_result_abv_thrsh.reshape(train_result_abv_thrsh.shape[0], 1)))
print quadratic_weighted_kappa(train_result, case_train)

param_grid = [
              {'risk': [1]}
             ]


# def opt_cut_v2_cv(x):
#     case = np.array(ranking(X_train, x)).astype('int')
#     score = -1 * quadratic_weighted_kappa(y_train, case)
#     return score
#
# print 'start CV'
# for params in ParameterGrid(param_grid):
#     print params
#     # CV
#     cv_n = 12
#     kf = StratifiedKFold(train_result, n_folds=cv_n, shuffle=True)
#     metric = []
#     for train_index, test_index in kf:
#         X_train, X_test = train_prediction[train_index], train_prediction[test_index]
#         y_train, y_test = train_result[train_index], train_result[test_index]
#         # train machine learning
#         res = optimize.minimize(opt_cut_v2_cv, [2.46039684, 3.48430979, 4.30777339, 4.99072484, 5.59295844,
#                                                 6.17412558, 6.79373477], method='Nelder-Mead',
#                                 # options={'disp': True}
#                                 )
#         # print res.x
#         cur_splitter = list(params['risk'] * res.x + (1 - params['risk']) * riskless_splitter)
#         # print cur_splitter
#         classified_predicted_results = np.array(ranking(X_test, cur_splitter)).astype('int')
#         # predict
#         print quadratic_weighted_kappa(y_test, classified_predicted_results)
#         metric.append(quadratic_weighted_kappa(y_test, classified_predicted_results))
#
#     print 'The quadratic weighted kappa is: ', np.mean(metric)
