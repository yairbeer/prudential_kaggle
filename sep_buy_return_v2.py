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

"""
preprocessing data
"""
need_dummying = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6',
                 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1',
                 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4',
                 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1',
                 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
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

train_result = trainset['Response']
# print train_result.value_counts()
# train_result.to_csv("train_result.csv")

trainset_cols = list(trainset.columns.values)
trainset = trainset.drop('Response', axis=1)

trainset = trainset.fillna('9999')

# trainset[need_dummying] = trainset[need_dummying].astype(str)
n = trainset.shape[0]

sparsity = 0.95
sparsity = n * (1 - sparsity)

print 'dummy train variables'
dummies = []
for var in need_dummying:
    print 'dummyfing trainset %s' % var
    # print trainset[var]
    dummy_col = pd.get_dummies(trainset[var])
    add_prefix(dummy_col, var + '_')
    dummies.append(dummy_col)
    trainset = trainset.drop(var, axis=1)

train = pd.concat([trainset] + dummies, axis=1)
# train = remove_sparse(train, sparsity * 0.25)

# preprocess test data
print 'read test data'
testset = pd.DataFrame.from_csv('test.csv', index_col=0)
testset = testset.fillna('9999')

testset[need_dummying] = testset[need_dummying].astype(str)
n_test = testset.shape[0]

sparsity = 0.95
sparsity = n_test * (1 - sparsity)

print 'dummy test variables'
dummies = []
for var in need_dummying:
    print 'dummyfing testset %s' % var
    dummy_col = pd.get_dummies(testset[var])
    add_prefix(dummy_col, var + '_')
    dummies.append(dummy_col)
    testset = testset.drop(var, axis=1)

test = pd.concat([testset] + dummies, axis=1)
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
train.to_csv("train_dummied.csv")
test.to_csv("test_dummied.csv")
