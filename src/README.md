## K-Nearest Neighbour

The nearest neighbor algorithm is used to find the k nearest neighbors of a specified point among a set of unstructured data points. 

## K-Means Clustering 

K-means clustering is a method of cluster analysis which aims to partition m instances into k clusters in which each instance belongs to the cluster with the nearest mean.

## Decision Tree Algorithm

A decision tree is a tool that is used for classification in machine learning, which uses a tree structure where internal nodes represent tests and leaves represent decisions. It makes use of information theoretic concepts such as entropy to classify the data.
- Choose the best attribute(s) to split the remaining instances and make that attribute a decision node
- Repeat this process for recursively for each child
- Stop when:
‣ All the instances have the same target attribute value
‣ There are no more attributes
‣ There are no more instances

### Dependencies
   1. numpy
   2. pandas


## Perceptron for Binary Classification

Perceptron is a linear classification model of binary classification, whose input is the feature vector of the instance and the output is the category of the instance.  It is used to define a hyperplane that divides the instance into positive and negative categories in the input space.

## Neural Networks for Classification.

We use keras to create a Neural Networks model. Keras is a popular Python package for deep neural networks with multiple backends, including TensorFlow. In this model, some basic data pre-processing techniques, such as scaler and encoding was used. It also contains Activation Functions and Hyper-parameters, etc.

### Dependencies
   1. Keras
   2. TensorFlow backend 
   3. sklearn
   4. pandas and numpy
 
 
## Genetic Programming for Classification/Regression

A symbolic regression is an estimator that begins by building a population of naive random formulas to represent a relationship. The formulas are represented as tree-like structures with mathematical functions being recursively applied to variables and constants. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations such as crossover, mutation or reproduction.

### Dependencies
   1. gplearn
   2. TensorFlow backend 
   3. sklearn
   4. pandas and numpy

The source code can be run under windows with python 3.7.6, pycharm and the libraries above.
