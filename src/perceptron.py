
import numpy as np
import pandas as pd

# load the data
data = pd.read_csv("C:/Users/DELL/Desktop/dataset.csv", header=0)
# data = pd.read_csv("C:/Users/DELL/Desktop/dataset.csv", header=None)
# # print(data.head(2))
x = data.values[:, :3]
y = data.values[:, -1]
# y = data.Class

# print(x)
print(y)
y = np.expand_dims(y, axis=1)
print(y)


# Set initial weight and bias
# w = np.mat([1.0, 1.0, 1.0])
bias = 0
w = np.mat(2*(np.random.random(3)-0.5))  # weights [-1,1]

# set learning rate
rate = 0.2


# Activation function
def sgn(x):
    return np.where(x > 0.0, 1, 0)


# Iterate 200 through learning rules
for i in range(200):
    # Error calculation
    errors = 0
    for j in range(len(x)):
        # Error calculation
        r = rate * (y[j] - sgn(x[j] * w.T + bias))
        # Adjust weight
        w += r * x[j]
        bias += r
        # Error calculation
        errors += abs(r)
    print(i, ' iter :error is ', errors)
    if errors == 0:
        break

print(sgn(x*w.T+bias))
print(y)
result = sgn(x*w.T+bias)
test_y = y

print('The accuracy of perceptron is：%.4f%%' % (100*np.sum(result == test_y) / len(result)))
print(w)