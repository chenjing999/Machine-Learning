#!/usr/bin/python
# coding=utf-8

from numpy import *
import pandas as pd
import numpy as np


# Load the data
def loadDataSet(fileName):
    dataSet = pd.read_csv('C:/Users/DELL/Desktop/part1/data/wine.csv')
    return dataSet


# Calculate Euclidean Distance np.sqrt(sum(a**2))
def calcEucliDist(point, centroid):
    return sqrt(sum(power(point - centroid, 2)))  # Find the distance between point and centroid


# Build cluster centers and take k random centroids
def randInitCent(dataSet, k):
    n = shape(dataSet)[1]
    print('The data set shape is：', dataSet.shape)
    # Initialize k n-dimensional centroids
    centroids = mat(zeros((k, n)))
    for j in range(n):
        # Find the minimum value and maximum values for each column(dimension)
        minJ = min(dataSet[:, j])
        # print('Minimum value in the %d column' % j, minJ)
        maxJ = max(dataSet[:, j])
        # print('Maximum value in the %d column:' % j, maxJ)
        rangeJ = float(maxJ - minJ)  # the value range of each dimension
        # print('range_j is:', rangeJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  # obtain the random number in each dimension
    print("The randomly selected initial mean vector is：\n", centroids)
    return centroids


# Implement k-means clustering algorithm
def kMeans(dataSet, k, distMeans=calcEucliDist, createCent=randInitCent):
    m = shape(dataSet)[0]
    print('m is:', m)
    """
    Initialize an all-zero array of m*2. Note that there are tuples inside
    Used to store the sample belongs to what kind and centroid distance
    clusterAssment: The first column stores the center point to which the data belongs,
    and the second column is the distance from the data to the center point.
    
    """
    clusterAssment = mat(zeros((m, 2)))
    # print('clusterAssment is：', clusterAssment)
    # step 1: init centroids
    # The random initial mean vector was found after the call
    centroids = createCent(dataSet, k)
    iter = 0
    # A convergence flag use to judge whether that cluster has converged
    converged = False
    while not converged and iter < 100:
        iter += 1
        converged = True
        # An ordered sequence from 0 to m sequentially takes out elements and assigns them to i
        for i in range(m):  # Divide each data point into its nearest center point.
            minDist = inf
            minIndex = -1
            # print('minDist is：', minDist)
            # step 2: find the centroid who is closest
            for j in range(k):
                # The initial mean vector array takes out the elements of two columns in each row
                # and the elements of two columns in all rows in the data set
                # to calculate the distance
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                # If the distance is the smallest, mark the row of the current mean vector.
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j  # If the i-th data point is closer to the j-th center point, i will be assigned to j.
            if clusterAssment[i, 0] != minIndex:
                converged = False     # If the centroid location changes, the iteration needs to continue.
            # Class and Euclidean Distance Square
            # and store the distribution of data point i in the dictionary
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        # step 4: update centroids
        for cent in range(k):
            ptsInClust = []
            for j in range(m):
                if clusterAssment[j, 0] == cent:
                    ptsInClust.append(dataSet[j].tolist()[0])
            ptsInClust = mat(ptsInClust)
            # print('ptsInClust is：', ptsInClust)
            # Average by column, i.e. new mean vector
            # Calculate the new centroid based on all samples in cluster cent
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    print('This is the %d iteration' % iter)
    """
     Returns the final stable centroid, as well as the centroid and 
    distance to which each sample belongs.
    """
    return centroids, clusterAssment


if __name__ == '__main__':
    # -----------------------------------test-------------------------------------
    # Using test data and test k-means algorithm
    dataSet = mat(loadDataSet('C:/Users/DELL/Desktop/wine.csv'))
    # print('randInitCent is：', randInitCent(dataSet, 5))
    myCentroids, clustAssing = kMeans(dataSet, 5)
    print('myCentroids is：\n', myCentroids)
    print('myCentroids shape is：', myCentroids.shape)
    # print('clustAssing[:, 0]--The categories of m rows of data are：\n', clustAssing[:, 0])
    print('clustAssing shape is: ', clustAssing.shape)
    # print(clustAssing)  # matrix  Centroid and distance to which each sample belongs
    a = clustAssing[:, 0]

    list = []  # Create an empty list
    for x in a.flat:
        list.append(int(x))  # Pass the result of loop c into the array
    print('The list of classes corresponding to all samples is：', list)

    myset = set(list)
    for item in myset:
        print("Cluster %d has %d instances" % (item, list.count(item)))


