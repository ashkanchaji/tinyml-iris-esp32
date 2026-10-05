import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

""" Phase 1 """

# Load dataset
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate mean and std for normalization
mean = X_train.mean(axis=0)
scale = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / scale
X_test_scaled = (X_test - mean) / scale

# Save scaling parameters
os.makedirs('models', exist_ok=True)
np.savez('models/iris_scaler_params.npz', mean=mean, scale=scale)

# Build neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_data=(X_test_scaled, y_test))

# Save Keras model
model.save('models/iris_model.keras')

""" Phase 2 """

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('models/iris_model.tflite', 'wb') as f:
    f.write(tflite_model)

""" Phase 3 """

# Convert TFLite model to C header
hex_lines = []
for i, val in enumerate(tflite_model):
    if i % 12 == 0:
        hex_lines.append('\n  ')
    hex_lines.append(f'0x{val:02x}, ')

c_content = f'''#ifndef IRIS_MODEL_H
#define IRIS_MODEL_H

const unsigned char iris_model_tflite[] = {{{"".join(hex_lines)}
}};
const unsigned int iris_model_tflite_len = {len(tflite_model)};

#endif
'''

os.makedirs('include', exist_ok=True)
with open('include/iris_model.h', 'w') as f:
    f.write(c_content)

print("Model training and header generation completed successfully.")
