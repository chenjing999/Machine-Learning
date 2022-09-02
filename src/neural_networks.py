import pandas
import random
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras as K
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback, EarlyStopping
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

SEED = 1234563839
random.seed(SEED)

# load the training data
data = pandas.read_csv("D:/comp2020/420/ass2_data/ass2_data/part2/wine-training.csv", encoding="latin-1")
print(data.head(5))
X = data.drop('Class', axis=1)
y = data['Class']
print(X)

X = X.values
y = y.values
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1000)
print(y[0])
# print('before changed %s' % x_train[0])

# load the test data
data1 = pandas.read_csv("D:/comp2020/420/ass2_data/ass2_data/part2/wine-test.csv", encoding="latin-1")
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
# print('after changed %s'% x_train[0])


num_classes = 3
y_train = K.utils.to_categorical(y_train - 1, num_classes)
y_val = K.utils.to_categorical(y_val - 1, num_classes)
y_test = K.utils.to_categorical(y1 - 1, num_classes)
print(y_test[6])
# print('The corresponding category is %s'% y_train[0])


# Define the model
# init = K.initializers.glorot_uniform(seed=1)
# simple_adam = K.optimizers.Adam()
# # hidden_layer_nodes = [1, 2, 3, 4, 5, 10, 20]
# model = Sequential()
# model.add(layers.Dense(units=5, input_dim=13, kernel_initializer=init, activation='relu'))
# model.add(layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
# model.summary()

# create model
model = Sequential()
model.add(Dense(5, input_dim=13, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='softmax'))

# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
#                              save_best_only=True, mode='auto')
# earlyStopping = EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')
# callbacks_list = [checkpoint, earlyStopping, metrics]

early_stopping = EarlyStopping(monitor='val_loss', patience=4)
# model.fit(X, y, validation_split=0.3, callbacks=[early_stopping])

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
b_size = 5
max_epochs = 20
print("Starting training ")
h = model.fit(x_train, y_train, batch_size=b_size, validation_data=(x_val, y_val),
              epochs=max_epochs, shuffle=True, verbose=1, callbacks=[early_stopping])
print("Training finished \n")

# evaluate model
print("Starting training ")
h = model.fit(x_train, y_train, batch_size=b_size,
              epochs=max_epochs, shuffle=True, verbose=False)
print("Training finished \n")

# eval = model.evaluate(x_test, y_test, verbose=0)
# print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
#           % (eval[0], eval[1] * 100) )

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

