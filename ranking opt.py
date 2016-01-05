import pandas as pd
import numpy as np
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
# print train_result
# print train_result['Response'].value_counts()
col = list(train_result.columns.values)
result_ind = list(train_result[col[0]].value_counts().index)
train_result = np.array(train_result).ravel()

train_prediction = pd.DataFrame.from_csv("meta_train_boost_regression.csv")
train_prediction = np.array(train_prediction).ravel()
base_splitter = np.arange(7) + 1.5
basecase = ranking(train_prediction, base_splitter)
# show = zip(list(basecase), list(train_prediction), list(train_result))
# for line in show[:1000]:
#     print line
# basecase_naive = train_prediction
print quadratic_weighted_kappa(train_result, basecase)

x0_range = np.arange(-5, 5, 0.5)
x1_range = np.arange(-1.5, 1.5, 0.25)
x2_range = np.arange(-0.5, 0.6, 0.1)
x3_range = np.arange(-0.5, 0.6, 0.1)
x4_range = np.arange(-0.5, 0.6, 0.1)
bestcase = np.array(ranking(train_prediction, base_splitter)).astype('int')
bestscore = quadratic_weighted_kappa(train_result, basecase)

print x0_range.shape[0] * x1_range.shape[0] * x2_range.shape[0] * x3_range.shape[0] * x4_range.shape[0]


def opt_cut(x):
    x0, x1, x2, x3, x4 = x
    case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter +
                                                               x2 * base_splitter**2 +
                                                               x3 * base_splitter**3 +
                                                               x4 * base_splitter**4))).astype('int')
    score = -quadratic_weighted_kappa(train_result, case)
    print score
    return score

res = optimize.minimize(opt_cut, [0.400000, 1.600000, -0.175000, 0.010000, 0], method='Nelder-Mead', options={'disp': True})
print res.x
x0, x1, x2, x3, x4 = res.x
case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter +
                                                               x2 * base_splitter**2 +
                                                               x3 * base_splitter**3 +
                                                               x4 * base_splitter**4))).astype('int')
print (x0 + x1 * base_splitter + x2 * base_splitter**2 + x3 * base_splitter**3 + x4 * base_splitter**4)
print quadratic_weighted_kappa(train_result, case)

# res = optimize.differential_evolution(opt_cut, [0.400000, 1.600000, -0.175000, 0.010000, 0], disp=True)
# print res.x
res = optimize.differential_evolution(opt_cut, [0, 1, 0, 0, 0], disp=True)
print res.x
x0, x1, x2, x3, x4 = res.x
case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter +
                                                               x2 * base_splitter**2 +
                                                               x3 * base_splitter**3 +
                                                               x4 * base_splitter**4))).astype('int')
print (x0 + x1 * base_splitter + x2 * base_splitter**2 + x3 * base_splitter**3 + x4 * base_splitter**4)
print quadratic_weighted_kappa(train_result, case)

# print 'start 4-th optimization'
# # optimize classifier
# n_iterations = 1000
# bestscore = 0
# for iteration in range(n_iterations):
#     x0, x1, x2, x3 = [0.400000, 1.600000, -0.175000, 0.010000]
#     # x0 = np.random.uniform(-1, 1)
#     # x1 = np.random.uniform(-1, 1)
#     # x2 = np.random.uniform(-0.2, 0.2)
#     # x3 = np.random.uniform(-0.2, 0.2)
#     x4 = np.random.uniform(-0.2, 0.2)
#     case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter +
#                                                x2 * base_splitter**2 +
#                                                x3 * base_splitter**3 +
#                                                x4 * base_splitter**4))).astype('int')
#     score = quadratic_weighted_kappa(train_result, case)
#     if score > bestscore:
#         bestscore = score
#         bestcase = case
#         print 'For splitter ', (x0 + x1 * base_splitter + x2 * base_splitter**2 +
#                                 x3 * base_splitter**3 + x4 * base_splitter**4)
#         print 'Variables x0 = %f, x1 = %f, x2 = %f, x3 = %f, x4 = %f' % (x0, x1, x2, x3, x4)
#         print 'The score is %f' % bestscore

# print 'start linear optimization'
# # optimize classifier
# for x0 in x0_range:
#     for x1 in x1_range:
#         case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter))).astype('int')
#         score = quadratic_weighted_kappa(train_result, case)
#         if score > bestscore:
#             bestscore = score
#             bestcase = case
#             print 'For splitter ', (x0 + x1 * base_splitter)
#             print 'Variables x0 = %f, x1 = %f' % (x0, x1)
#             print 'The score is %f' % bestscore
#
#
# print 'start quadratic optimization'
# # optimize classifier
# for x0 in x0_range:
#     for x1 in x1_range:
#         for x2 in x2_range:
#             case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter + x2 * base_splitter**2))).astype('int')
#             score = quadratic_weighted_kappa(train_result, case)
#             if score > bestscore:
#                 bestscore = score
#                 bestcase = case
#                 print 'For splitter ', (x0 + x1 * base_splitter + x2 * base_splitter**2)
#                 print 'Variables x0 = %f, x1 = %f, x2 = %f' % (x0, x1, x2)
#                 print 'The score is %f' % bestscore
#
# print 'start cubic optimization'
# # optimize classifier
# for x0 in x0_range:
#     for x1 in x1_range:
#         for x2 in x2_range:
#             for x3 in x3_range:
#                 case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter + x2 * base_splitter**2 +
#                                                            x3 * base_splitter**3))).astype('int')
#                 score = quadratic_weighted_kappa(train_result, case)
#                 if score > bestscore:
#                     bestscore = score
#                     bestcase = case
#                     print 'For splitter ', (x0 + x1 * base_splitter + x2 * base_splitter**2 +
#                                                            x3 * base_splitter**3)
#                     print 'Variables x0 = %f, x1 = %f, x2 = %f, x3 = %f' % (x0, x1, x2, x3)
#                     print 'The score is %f' % bestscore

# basecase_naive baseline: 0.609289171153
# 4th
# x = [  4.17220587e-01   1.60267961e+00  -1.75452819e-01   1.00609841e-02  -5.95422308e-06]
# [ 2.46039684  3.48430979  4.30777339  4.99072484  5.59295844  6.17412558 6.79373477]
# 0.658710095337

# cubic
# For splitter  [ 2.44    3.4625  4.285   4.9675  5.57    6.1525  6.775 ]
# Variables x0 = 0.400000, x1 = 1.600000, x2 = -0.175000, x3 = 0.010000
# The score is 0.658157

# quad
# For splitter  [ 2.28125  3.38125  4.33125  5.13125  5.78125  6.28125  6.63125]
# Variables x0 = 0.350000, x1 = 1.400000, x2 = -0.075000
# The score is 0.655544

# lin
# For splitter  [ 1.85  2.75  3.65  4.55  5.45  6.35  7.25]
# Variables x0 = 0.500000, x1 = 0.900000
# The score is 0.632679