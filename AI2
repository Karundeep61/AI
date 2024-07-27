import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Verify GPU setup
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
target = iris.target

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Convert to float32
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

# Create a tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Define the model using the high-level API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=7, validation_data=test_dataset)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
