import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

data = pd.read_csv("dataset/traces.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = X.reshape(X.shape[0], X.shape[1], 1)

y = to_categorical(y, 256)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    Conv1D(
        32,
        kernel_size=3,
        activation='relu',
        input_shape=(100,1)
    )
)

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(256, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

model.save("model/cnn_model.keras")

print("Model Saved")