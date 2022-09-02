import pandas
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras as K
from sklearn.neighbors import KNeighborsClassifier

# Set random seeds to ensure the reproducible results
# SEED = 500
# random.seed(SEED)


# load the training data
data = pandas.read_csv("D:/comp2020/420/ass2_data/ass2_data/part2/wine-training.csv",encoding="latin-1")
print(data.head(5))
X = data.drop('Class', axis=1)
y = data['Class']
print(X)

X = X.values
y = y.values
# x_train, y_train = X, y
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1000)
print(y[0])

# load the test data
data1 = pandas.read_csv("D:/comp2020/420/ass2_data/ass2_data/part2/wine-test.csv",encoding="latin-1")
data1.head(5)
X1 = data1.drop('Class', axis=1).values
y1 = data1['Class'].values
print(X1.shape)


scaler = StandardScaler()
# Fit only to the training data
print(scaler.fit(x_train))
# Now apply the transformations to the data:

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(X1)

num_classes = 3
y_train = K.utils.to_categorical(y_train-1, num_classes)
y_val = K.utils.to_categorical(y_val-1, num_classes)
y_test = K.utils.to_categorical(y1-1, num_classes)
print(y_test[6])


# KNN model
knn_model = KNeighborsClassifier(n_neighbors=num_classes)
knn_model.fit(x_train, y_train)
accuracy = knn_model.score(x_train, y_train)
print('The KNN accuarcy on train dataset:{:.2f}%'.format(accuracy * 100))
# accuracy = knn_model.score(x_val, y_val)
# print('The accuarcy on validation dataset:{:.2f}%'.format(accuracy * 100))
accuracy = knn_model.score(x_test, y_test)
print('The KNN accuarcy on test dataset:{:.2f}%'.format(accuracy * 100))
