import pandas as pd
import numpy as np

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

# preprocess test data
print 'read train data'
trainset = pd.DataFrame.from_csv('train.csv', index_col=0)
train_result = trainset['Response']

# print train_result.value_counts()
# train_result.to_csv("train_result.csv")

trainset_cols = list(trainset.columns.values)
trainset = trainset.drop('Response', axis=1)
trainset = trainset.fillna('999')

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('test.csv', index_col=0)
testset = testset.fillna('999')


# special dummies
train_p_info2_dummy = dummy_2d(trainset, 'Product_Info_2')
trainset = trainset.drop('Product_Info_2', axis=1)
test_p_info2_dummy = dummy_2d(testset, 'Product_Info_2')
testset = testset.drop('Product_Info_2', axis=1)

# trainset[need_dummying] = trainset[need_dummying].astype(str)
n = trainset.shape[0]
# testset[need_dummying] = testset[need_dummying].astype(str)
n_test = testset.shape[0]

# sparsity = 0.95
# sparsity = n * (1 - sparsity)
# sparsity = n_test * (1 - sparsity)

# Plotting

train = pd.concat([trainset, train_p_info2_dummy], axis=1)
# train = remove_sparse(train, sparsity * 0.25)

test = pd.concat([testset, test_p_info2_dummy], axis=1)
# test = remove_sparse(test, sparsity * 0.25)

# Find common coloumns
col_train = list(train.columns.values)
print col_train
col_test = list(test.columns.values)
print col_test
col_common = []
# add only common columns for train and test
for col in col_train:
    if col in col_test:
        col_common.append(col)
print col_common

train = train[col_common]
test = test[col_common]

print 'write to data'
train.to_csv("train_v2.csv")
test.to_csv("test_v2.csv")
