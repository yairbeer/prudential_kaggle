import pandas as pd
import numpy as np
import scipy.stats as stats

__author__ = 'WiBeer'


def add_prefix(dataset, prefix):
    col_names = list(np.array(dataset.columns.values).astype('str'))
    for i_in in range(len(col_names)):
        col_names[i_in] = prefix + col_names[i_in]
    dataset.columns = col_names
    return dataset


def remove_sparse(dataset, n_valid_samples):
    col_names = list(dataset.columns.values)
    dead_cols = []
    for i_in in range(len(col_names)):
        if np.abs(np.sum(np.array(dataset[col_names[i_in]]))) < n_valid_samples:
            dead_cols.append(col_names[i_in])

    dataset = dataset.drop(dead_cols, axis=1)
    return dataset


def dummy_2d(dataset, column):
    print '2d dummying %s' % column
    index = dataset.index
    dataset = list(dataset[column].astype('str'))
    dataset_1 = map(lambda x: x[0], dataset)
    dataset_2 = map(lambda x: x[1], dataset)
    dataset_1 = np.array(dataset_1).reshape((len(dataset_1), 1))
    dataset_2 = np.array(dataset_2).reshape((len(dataset_2), 1))
    dataset = pd.get_dummies(pd.DataFrame(np.hstack((dataset_1, dataset_2))))
    dataset.index = index
    dataset = add_prefix(dataset, column)
    # print dataset
    return dataset


def dummy_num(dataset, column):
    print 'num dummying %s' % column
    index = dataset.index
    dataset = list(dataset[column].astype('str'))
    # Find max digits
    n_dig = 0
    for i in range(len(dataset)):
        if len(dataset[i]) > n_dig:
            n_dig = len(dataset[i])
    # Fill digits
    for i in range(len(dataset)):
        if len(dataset[i]) < n_dig:
            for j in range(n_dig - len(dataset[i])):
                dataset[i] = '0' + dataset[i]
    dataset_digits = []
    for i in range(n_dig):
        dataset_digits.append(map(lambda x: x[i], dataset))
        dataset_digits[i] = np.array(dataset_digits[i]).reshape((len(dataset_digits[i]), 1))
    dataset = pd.get_dummies(pd.DataFrame(np.hstack(tuple(dataset_digits))))
    dataset.index = index
    dataset = add_prefix(dataset, column)
    # print dataset
    return dataset


"""
preprocessing data
"""
need_dummying = ['Product_Info_1', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1',
                 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4',
                 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
                 'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
                 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
                 'Medical_History_10', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13',
                 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
                 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22',
                 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27',
                 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
                 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
                 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40',
                 'Medical_History_41']


# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('train.csv', index_col=0)
train_result = np.array(trainset['Response']).ravel()

print 'read test data'
testset = pd.DataFrame.from_csv('test.csv', index_col=0)
# print train_result.value_counts()
# train_result.to_csv("train_result.csv")

trainset_cols = list(trainset.columns.values)
trainset = trainset.drop('Response', axis=1)
trainset = trainset.fillna('999')

# trainset[need_dummying] = trainset[need_dummying].astype(str)
n = trainset.shape[0]
# testset[need_dummying] = testset[need_dummying].astype(str)
n_test = testset.shape[0]

"""
Correlation
"""
# plottable = ['Product_Info_3', 'Employment_Info_2', 'InsuredInfo_3', 'Medical_History_10']
# for var in plottable:
#     print var
#     print 'pearson cor: ', stats.pearsonr(np.array(trainset[var]), train_result)
#     print 'spearman cor: ', stats.spearmanr(np.array(trainset[var]), train_result)
# nope


"""
Better dummying
"""
dummies = []
dummies_test = []
for var in need_dummying:
    print 'dummyfing %s' % var

    print trainset[var].value_counts()
    dummy_col = pd.get_dummies(trainset[var])
    add_prefix(dummy_col, var + '_')
    dummy_col.index = trainset.index
    trainset = trainset.drop(var, axis=1)

    print testset[var].value_counts()
    dummy_test_col = pd.get_dummies(testset[var])
    add_prefix(dummy_test_col, var + '_')
    dummy_test_col.index = testset.index
    testset = testset.drop(var, axis=1)

    # Find common coloumns
    col_train = list(dummy_col.columns.values)
    col_test = list(dummy_test_col.columns.values)
    col_common = []
    col_train_only = []
    # add only common columns for train and test
    for col in col_train:
        if col in col_test:
            col_common.append(col)
        else:
            col_train_only.append(col)

    dummy_col = dummy_col[col_common]
    dummy_test_col = dummy_test_col[col_common]
    dummies.append(dummy_col)
    dummies_test.append(dummy_test_col)

    if col_train_only:
        print col_common
        print col_train_only
