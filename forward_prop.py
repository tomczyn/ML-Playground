import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from lab_coffee_utils import load_coffee_data

X, Y = load_coffee_data()
print(X.shape, Y.shape)

print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000, 1))
print(Xt.shape, Yt.shape)

tf.random.set_seed(1234)  # applied to achieve consistent results. Same seed same rando numbers
model = Sequential(
    [
        # Shape is dimensions for the input, two variables per row is a 2 dimension shape
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name='layer1'),
        Dense(1, activation='sigmoid', name='layer2')
    ]
)
model.summary()
L1_num_params = 2 * 3 + 3  # W1 parameters + b1 parameters
L2_num_params = 3 * 1 + 1  # W2 parameters + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# Defines a loss function and specifies compile optimization
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

# Runs gradient descent and fits the weights to the data
model.fit(
    Xt, Yt,
    epochs=10,
)

# Updated weights
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

X_test = np.array([
    [200, 13.9],  # Positive example
    [200, 17]  # Negative example
])
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("Predictions = \n", predictions)

# Apply threshold to get a decision
yhat = (predictions >= 0.5).astype(int)
print(f"Decisions = \n{yhat}")
