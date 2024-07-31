import sys
import io
import tensorflow as tf
import tensorflow_datasets as tfds
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

# Load the EMNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Normalize the data
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(32)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

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