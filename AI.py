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

# Load the EMNIST dataset from local files
def load_emnist_data(image_path, label_path):
    with open(image_path, 'rb') as imgpath, open(label_path, 'rb') as lblpath:
        imgpath.read(16)  # skip the header
        lblpath.read(8)   # skip the header
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(-1, 28, 28, 1)
        labels = np.frombuffer(lblpath.read(), dtype=np.uint8)
    return images, labels

train_images, train_labels = load_emnist_data(
    'F:/Git/vscodeAI/AI/Data/emnist-balanced-train-images-idx3-ubyte',
    'F:/Git/vscodeAI/AI/Data/emnist-balanced-train-labels-idx1-ubyte'
)
test_images, test_labels = load_emnist_data(
    'F:/Git/vscodeAI/AI/Data/emnist-balanced-test-images-idx3-ubyte',
    'F:/Git/vscodeAI/AI/Data/emnist-balanced-test-labels-idx1-ubyte'
)

# Normalize the data
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Convert to TensorFlow datasets
ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),   # Flatten the input image
    tf.keras.layers.Dense(128, activation='relu'),      # Hidden layer with 128 nodes
    tf.keras.layers.Dense(47, activation='softmax')     # Output layer for 47 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(ds_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(ds_test)
print('\nTest accuracy:', test_acc)

# Use new data for predictions
for images, labels in ds_test.take(1):
    x_new = images.numpy()
    y_new = labels.numpy()

# Visualize the new input data
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_new[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# Make predictions
predictions = model.predict(x_new[:10])

# Get the predicted class for each input
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted and actual classes
print("Predicted classes:", predicted_classes)
print("Actual classes:   ", y_new[:10])
