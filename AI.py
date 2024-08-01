import sys
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Redirect standard output and error to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU device found:', tf.test.gpu_device_name())
else:
    print("No GPU found. Running on CPU.")

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model with more nodes
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # Flatten the input image
    tf.keras.layers.Dense(256, activation='relu'),   # Hidden layer with 256 nodes
    tf.keras.layers.Dense(128, activation='relu'),   # Hidden layer with 128 nodes
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with more epochs
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# Plot the loss and accuracy graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation accuracy values
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Make predictions
predictions = model.predict(x_test[:10])

# Get the predicted class for each input
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted and actual classes
print("Predicted classes:", predicted_classes)
print("Actual classes:   ", y_test[:10])

# Visualize the new input data and predictions
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}")
    plt.axis('off')
plt.show()
