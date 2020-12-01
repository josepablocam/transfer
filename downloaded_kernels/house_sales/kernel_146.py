import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import numpy as np

df = pd.read_csv('../input/kc_house_data.csv')
dataset = df.values

# Split the dataset into X and Y
X = dataset[:,3:]
X = np.array(X, dtype='float')
Y = dataset[:,2]
Y = np.array(Y, dtype='float')

# Choose a train size, and split the data into a train and test set.
train_size = int(0.8 * X.shape[0])
X_train = X[:train_size,:]
Y_train = Y[:train_size]

# Compute the mean and standard deviation of Y_train, so that we can normalise the data.
Y_train_mean = np.mean(Y_train)
Y_train_std = np.std(Y_train)
Y_train = (Y_train - Y_train_mean) / Y_train_std

# Do the same for X_train
X_train_means = np.mean(X_train, axis=0)
X_train_stds = np.std(X_train, axis=0)
X_train = (X_train - X_train_means) / X_train_stds

# Also compute the test set, and normalise it using the training mean and standard deviations
X_test = X[train_size:,:]
Y_test = Y[train_size:]
X_test = (X_test - X_train_means) / X_train_stds
Y_test = (Y_test - Y_train_mean) / Y_train_std

# Build the model.
model = Sequential()
model.add(Dense(100, input_dim=18, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, input_dim=18, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, init='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
early_stopping = EarlyStopping(patience=10)
model.fit(X_train, Y_train, nb_epoch=1000, batch_size=256, verbose=2,
        validation_split=0.2, callbacks=[early_stopping])
test_loss = model.evaluate(X_test, Y_test)

# The test loss is given both in normalised and in $ values
print("Test loss", test_loss, test_loss * Y_train_std)
Y_pred = model.predict(X_test)
Y_diff = Y_train_std*(np.ravel(Y_pred) - np.ravel(Y_test))
print(np.mean( Y_diff**2 ))