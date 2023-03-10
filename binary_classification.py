import warnings

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from autils import *

warnings.simplefilter(action='ignore', category=FutureWarning)

tf.autograph.set_verbosity(0)

X, y = load_data()

model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(25, activation='sigmoid', name='layer1'),
        Dense(15, activation='sigmoid', name='layer2'),
        Dense(1, activation='sigmoid', name='layer3'),
    ], name='binary_classification'
)

model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X, y,
    epochs=20
)

prediction = model.predict(X[0].reshape(1, 400))  # data for zero
print(f"predicting a zero: ${prediction}")
prediction = model.predict(X[500].reshape(1, 400))  # data for one
print(f"predicting a one: ${prediction}")

if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print(f"prediction after threshold: {yhat}")


# Custom naive implementation of Dense layer
def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out


# Custom vectorized (optimized) implementation of Dense layer
def my_dense_vectorized(A_in, W, b, g):
    z = np.matmul(A_in, W) + b
    A_out = g(z)
    return A_out
