import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('dataset.csv')

df_arr = np.array(df)
random.shuffle(df_arr)

n_features = 3

ds = df_arr[:, :n_features]
scaler = MinMaxScaler()
ds = scaler.fit_transform(ds).astype('float32')

X = ds
y = df_arr[:, n_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = keras.Sequential()
model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=0.5), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)

# Test 1
loss, acc = model.evaluate(X_test, y_test, verbose=2)

# Test 2
predictions = model.predict(X_test)
predictions = np.array([(1 if x > 0.5 else 0) for x in predictions])
print(predictions)
acc_score = accuracy_score(y_test, predictions)
rec_score = recall_score(y_test, predictions, average=None, zero_division=0)
print("acc = %.5f" % acc_score, "rec = [%.5f, %.5f]" % (rec_score[0], rec_score[1]))
print(sum(y_test)/len(y_test))
