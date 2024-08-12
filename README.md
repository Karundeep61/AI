# AI
 Neural Networks
Detailed Explanation of the Script

import sys
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
Importing Libraries: This script uses the sys, io, tensorflow, numpy, and matplotlib.pyplot libraries. These are essential for data manipulation, building and training the neural network, and plotting the results.

# Redirect standard output and error to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
Redirecting Output: This part ensures that the standard output and error streams use UTF-8 encoding, which helps to handle any special characters properly.

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU device found:', tf.test.gpu_device_name())
else:
    print("No GPU found. Running on CPU.")
Checking for GPU: TensorFlow can leverage GPUs for faster computation. This section checks if a GPU is available and prints its name; otherwise, it indicates that the code will run on the CPU.

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
Loading the Dataset: The MNIST dataset is loaded, which consists of handwritten digit images. The dataset is split into training and testing sets.
python
Copy code
# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0
Data Normalization: The pixel values of the images are normalized to a range of 0 to 1 by dividing by 255.0. This helps in faster convergence during training.

# Define the model with more nodes
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # Flatten the input image
    tf.keras.layers.Dense(256, activation='relu'),   # Hidden layer with 256 nodes
    tf.keras.layers.Dense(128, activation='relu'),   # Hidden layer with 128 nodes
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])
Defining the Model: A Sequential model is defined with three layers:
Flatten Layer: Converts the 2D image into a 1D array.
Dense Layers: Fully connected layers with 256 and 128 neurons using ReLU activation function.
Output Layer: A dense layer with 10 neurons (one for each digit) using the softmax activation function to output probabilities.

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
Compiling the Model: The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. The accuracy metric is used to evaluate the model's performance.

# Train the model with more epochs
history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
Training the Model: The model is trained for 15 epochs using the training data. The validation data (testing set) is used to evaluate the model after each epoch.

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)
Evaluating the Model: The model's performance is evaluated on the test data, and the test accuracy is printed.

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
Plotting Training Progress: The training and validation accuracy and loss are plotted to visualize how the model's performance improves over the epochs.

# Make predictions
predictions = model.predict(x_test[:10])

# Get the predicted class for each input
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted and actual classes
print("Predicted classes:", predicted_classes)
print("Actual classes:   ", y_test[:10])
Making Predictions: The model predicts the classes for the first 10 test images. The predicted and actual classes are printed.

# Visualize the new input data and predictions
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {predicted_classes[i]}")
    plt.axis('off')
plt.show()
Visualizing Predictions: The first 10 test images are displayed with their predicted classes.
