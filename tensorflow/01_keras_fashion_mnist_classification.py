# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)



# Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset
# often used as the "Hello, World" of machine learning programs for computer vision. 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\ntraining data:")
print(train_images.shape)
print(len(train_labels))
print(train_labels)

print("\ntesting data:")
print(test_images.shape)
print(len(test_labels))
print(test_labels)

#%matplotlib inline

# Display the first 25 images from the training set and display the class name below each image.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# preprocess the data
# scale these values to a range of 0 to 1 before feeding to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Dense(16, activation=tf.nn.relu), # These are densely-connected, or fully-connected, neural layers w/16 nodes (or neurons).
    keras.layers.Dense(10, activation=tf.nn.softmax) # The second layer is a 10-node softmax layer — this returns an array of 10 probability scores that sum to 1
])
model.compile(
    optimizer=tf.train.AdamOptimizer(), # This is how the model is updated based on the data it sees and its loss function.
    loss='sparse_categorical_crossentropy', # measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
    metrics=['accuracy']) # Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# Make predictions
print('\nMake predictions:')
predictions = model.predict(test_images)
print(predictions[0])
print("Prediction: " + str(np.argmax(predictions[0])) )
print("Correct answer: " + str(test_labels[0]))



