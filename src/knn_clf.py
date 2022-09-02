# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/DELL/Desktop/part1/data/wine-training.csv", header=0)
data2 = pd.read_csv("C:/Users/DELL/Desktop/part1/data//wine-test.csv", header=0)
print(data.head(2))
X_train = data.drop('Class', axis=1)
y_train = data.Class
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

test_X2 = data2.drop('Class', axis=1)
test_y2 = data2.Class


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        result = []
        for x in X:
            # step 1: calculate Euclidean distance
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            # step 2: sort the distance
            # argsort() returns the indices that would sort an array in a ascending order
            index = dis.argsort()
            # step 3: choose the min k distance
            index = index[:self.k]
            # print(index)  # for example, the min k distance[20 34 12]
            # print(self.y[index]) # [3 3 2]
            # print(dis[index]) # [11.61047372 20.73670659 29.33154445]
            # count = np.bincount(self.y[index], weights=1 / dis[index])
            # dis[index]=[11.61047372,20.73670659,29.33154445]
            # count = np.bincount([3,3,2],weights=1 / dis[index])
            # step 4: count the times labels occur
            count = np.bincount(self.y[index].astype(np.int32), weights=1 / (1 + dis[index]))
            # print(count) # return each elements, which is the times index occurs
            # step 5: the max voted class will return
            result.append(count.argmax())
        return np.asarray(result)


# when k=1
knn = KNN(k=1)
knn.fit(train_X, train_y)
result = knn.predict(test_X2)

print('There are %d instances in test data set.' % len(result))
print('There are %d results calculated by KNN1 are equal to the test data set.' % np.sum(result == test_y2))
print('The accuracy of KNN1 is：%.4f%%' % (100*np.sum(result == test_y2) / len(result)))
print('KNN1 predicts the class label of each instance: %s' % result)  # [2 2 2 1 2 1 2 2 2 1 1 3 3 2 2 3 3 2]

# # when k=3
# knn3 = KNN(k=3)
# knn3.fit(train_X, train_y)
# result3 = knn3.predict(test_X2)
#
# print('There are %d instances in test data set.' % len(result3))
# print('There are %d results calculated by KNN3 are equal to the test data set.' % np.sum(result3 == test_y2))
# print('The accuracy of KNN3 is：%.4f%%' % (100*np.sum(result3 == test_y2) / len(result3)))
# print('KNN3 predicts the class labels of each instance: %s' % result3) # [2 2 2 1 2 1 2 2 2 1 1 3 3 2 2 3 3 2]





