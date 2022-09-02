#!/usr/bin/python
# coding=utf-8
# Reference: https://github.com/gitkeidy/testDecisionTree/blob/master/tree.py

import numpy as np
import pandas as pd


# load the data
def loadDataSet():
    dataSet = pd.read_csv("C:/Users/DELL/Desktop/part2/data/hepatitis-training.csv", header=0)
    """10 other pairs training and test data set"""
    # dataSet = pd.read_csv("C:/Users/DELL/Desktop/data420/hepatitis-training-run-0.csv", header=0)
    return dataSet


# calculate the entropy
def calculateEntropy(dataSet):
    n = dataSet.shape[0]  # Get number of samples: 112
    print('n is %d' % n)
    label_type = dataSet.iloc[:, -1].value_counts()  # All categories of tags
    print('class_type is %s' % label_type)
    p = label_type / n  # #Probability of each category, i.e. p (Ci)
    # print('p is %s' % p)
    entropy = (-p * np.log2(p)).sum()
    # print('entropy is %s' % entropy)
    return entropy


# Select the best column/feature for split based on information gain
def chooseBestFeature(dataSet):
    base_entropy = calculateEntropy(dataSet)  # Calculate the original data set entropy
    bestGain = 0  # Initial info gain
    axis = -1  # Initialize the best feature index
    for i in range(dataSet.shape[1] - 1):  # Traverse each column of features
        featValue = dataSet.iloc[:, i].value_counts().index  # get the value of feature
        child_entropy = 0  # Initial child entropy
        for j in featValue:
            childSet = dataSet[dataSet.iloc[:, i] == j]  # Child node value distribution
            entropy = calculateEntropy(childSet)  # child node entropy
            child_entropy += (childSet.shape[0] / dataSet.shape[0]) * entropy  # weighted sum current column entropy
        # print(f'The{i}column entropy is{child_entropy}')
        infoGain = base_entropy - child_entropy  # current column info gain
        # print(f'The{i}column info gain is{infoGain}')
        if infoGain > bestGain:
            bestGain = infoGain  # choose the max info gain
            axis = i  # The max info gain feature/column index
    print("The best feature index is：%s" % axis)
    return axis


# Divides data sets based on chosen columns and feature value
def splitDataset(dataSet, axis, value):
    col = dataSet.columns[axis]  # best feature/column index name
    # split data set and delete current optimal feature
    newDataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return newDataSet


# Build the tree
def createTree(dataSet):
    featList = list(dataSet.columns)  # list all the columns
    classlist = dataSet.iloc[:, -1].value_counts()  # get the last column/class label
    # Judge whether the maximum number of labels is equal to the number of rows in the data set
    # or whether the data set has only one column
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]  # return the class label if true
    axis = chooseBestFeature(dataSet)  # determine the best split column index
    bestFeat = featList[axis]  # get the feature corresponding to the index
    tree = {bestFeat: {}}  # save the tree by dictionary
    del featList[axis]  # delete the current feature
    valueList = set(dataSet.iloc[:, axis])  # Select the best split to list all attribute values
    for value in valueList:  # Recursive build tree for each attribute Value
        tree[bestFeat][value] = createTree(splitDataset(dataSet, axis, value))
    return tree


# classify for the instance
# labels uses to store the best feature tag
def classify(inputTree, labels, testVec):
    firstStr = next(iter(inputTree))  # get the first node of the generated tree
    secondDict = inputTree[firstStr]  # next dictionary
    featIndex = labels.index(firstStr)  # the index of column in first node
    for key in secondDict.keys():
        if testVec[featIndex] == key:   # classify by feature
            if type(secondDict[key]) == dict:  # continue if the result is dict
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]  # stop if the result is not dict
    return classLabel


# predict the classification
def predictAccuracy(train, test):
    inputTree = createTree(train)
    labels = list(train.columns)  # all the columns names
    result = []
    for i in range(test.shape[0]):  # test each instance in test set
        testVec = test.iloc[i, :-1]  # instance in test set
        classLabel = classify(inputTree, labels, testVec)  # predict the class of instance
        result.append(classLabel)  # append classification results to the list
    test = test.copy()
    test['predict'] = result  # append the prediction to the last column of the test set
    accuracy = (test.iloc[:, -1] == test.iloc[:, -2]).mean()
    print(f'The accuracy of decision tree model is %0.2f:' % accuracy)
    return test


if __name__ == '__main__':

    dataSet = loadDataSet()
    print('----------------------')
    print(dataSet)
    print('***********************')
    print(calculateEntropy(dataSet))
    print('#######################')
    print(chooseBestFeature(dataSet))
    print('=======================')
    print(splitDataset(dataSet, 0, 1))

    # save to txt file
    doc = open('tree.txt', 'a')
    tree = createTree(dataSet)
    # print(tree)
    print(tree, file=doc)
    doc.close()

    train = dataSet
    test = pd.read_csv("C:/Users/DELL/Desktop/part2/data/hepatitis-test.csv", header=0)
    """10 other pairs training and test data set"""
    # test = pd.read_csv("C:/Users/DELL/Desktop/data420/hepatitis-test-run-0.csv", header=0)
    print(predictAccuracy(train, test))

