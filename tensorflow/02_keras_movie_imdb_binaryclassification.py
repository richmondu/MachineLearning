# Text classification with movie reviews
# This classifies movie reviews as positive or negative using the text of the review.
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)



# the IMDB dataset that contains the text of 50,000 movie reviews from the Internet Movie Database
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]))
print(len(train_data[1]))

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_data[0]))

# Prepare the data
# the movie reviews must be the same length, we will use the pad_sequences function to standardize the lengths:
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
print(len(train_data[0]))
print(len(train_data[1]))
print(train_data[0])


# Build the model
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
model = keras.Sequential()
# The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. 
#These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
model.add(keras.layers.Embedding(vocab_size, 16))
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. 
# This allows the model to handle input of variable length, in the simplest way possible.
model.add(keras.layers.GlobalAveragePooling1D())
# This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# The last layer is densely connected with a single output node. 
# Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability, or confidence level.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
# Since this is a binary classification problem and the model outputs a probability (a single-unit layer with a sigmoid activation), 
# we'll use the binary_crossentropy loss function.
# This isn't the only choice for a loss function, you could, for instance, choose mean_squared_error. 
# But, generally, binary_crossentropy is better for dealing with probabilities
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy'])


# Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('\nTest accuracy:', test_acc)

# Make predictions
print('\nMake predictions:')
predictions = model.predict(test_data)
print(predictions[0])
print("Prediction: " + str(np.argmax(predictions[0])) )
print("Correct answer: " + str(test_labels[0]))
print("\n" + decode_review(test_data[0]))


# Create a graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()
dict_keys(['acc', 'val_acc', 'loss', 'val_loss'])
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
