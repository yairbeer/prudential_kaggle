from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboostlib.xgboost as xgboost
from sklearn.cross_validation import StratifiedKFold

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

# basecase_naive baseline: -0.00590368902887
# add constant
x0_range = np.arange(-0.5, 0.55, 0.05)
x1_range = np.arange(0.1, 1.35, 0.05)
bestcase = np.array(ranking(train_prediction, base_splitter)).astype('int')
bestscore = quadratic_weighted_kappa(train_result, basecase)

print 'start linear optimization'
# optimize classifier
for x0 in x0_range:
    for x1 in x1_range:
        case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter))).astype('int')
        score = quadratic_weighted_kappa(train_result, case)
        if score > bestscore:
            bestscore = score
            bestcase = case
            print 'For splitter ', (x0 + x1 * base_splitter)
            print 'The score is %f' % bestscore

# add constant
x0_range = np.arange(-0.5, 0.55, 0.05)
x1_range = np.arange(0.1, 1.35, 0.05)
x2_range = np.arange(-0.5, 0.525, 0.025)
bestcase = np.array(ranking(train_prediction, base_splitter)).astype('int')
bestscore = quadratic_weighted_kappa(train_result, basecase)

print 'start quadratic optimization'
# optimize classifier
for x0 in x0_range:
    for x1 in x1_range:
        for x2 in x2_range:
            case = np.array(ranking(train_prediction, (x0 + x1 * base_splitter + x2 * base_splitter**2))).astype('int')
            score = quadratic_weighted_kappa(train_result, case)
            if score > bestscore:
                bestscore = score
                bestcase = case
                print 'For splitter ', (x0 + x1 * base_splitter)
                print 'The score is %f' % bestscore

# quad
# For splitter  [ 1.5   2.25  2.8   3.15  3.3   3.25  3.  ]
# The score is -0.006518

# lin
# For splitter  [ 1.85  2.75  3.65  4.55  5.45  6.35  7.25]
# The score is 0.632679