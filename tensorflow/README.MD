https://www.tensorflow.org/tutorials/keras/
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/index.md

# Learn and Use ML

01_keras_fashion_mnist_classification.py
- Basic image classification using Fashion MNIST dataset

02_keras_movie_imdb_binaryclassification.py
- Movie review binary classification (positive or negative) using IMDB dataset

03_keras_fuel_autompg_regression.py
- Predict fuel efficiency of late-1970s and early 1980s automobiles using Auto MPG Dataset

04_keras_l2regularization_droput_preventoverfitting.py
- L2 regularization and dropout to prevent overfitting

05_keras_save_and_restore_models.py
- save weights only or weights+configurations+optimizers

06_eager_execution_tensors.py
- tensors basics

07_eager_automatic_differentiation.py



# Build the model
    //cp_callback = tf.keras.callbacks.ModelCheckpoint("training_1/cp.ckpt", save_weights_only=True, verbose=1)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(x, y))) // transforms format to a 1-d array
    model.add(keras.layers.Dense(num_nodes, activation))   // densely-connected, or fully-connected
    //model.add(keras.layers.Dropout(0.5)) // dropout for regularization (prevent overfitting)
    //model.add(keras.layers.Dense(num_nodes, activation, kernel_regularizer=keras.regularizers.l2(0.001))) // l2 regularization (prevent overfitting)
    model.summary()
    model.compile(optimizer, loss, metrics)

# Train the model or load a saved model
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) *** Early stopping
    model.fit(train_images, train_labels, epochs=5, batch_size=512, callbacks=[cp_callback, early_stop, PrintDot()]) *** Checkpoint   callback, Early stopping callback
    //model.load_weights(checkpoint_path) // load a saved checkpoint model instead of train/fit

# Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    loss, mae, mse = model.evaluate(normed_test_data, test_labels)

# Make predictions
    predictions = model.predict(test_images)
    print("Prediction: " + str(np.argmax(predictions[0])) )



# Activations:
1. tf.nn.relu
2. tf.nn.softmax - returns probability scores that sum to 1
3. tf.nn.sigmoid - returns float between 0 and 1, representing a probability, or confidence level

# Optimizers:
1. tf.train.AdamOptimizer
2. tf.train.RMSPropOptimizer

# Loss functions:
1. sparse_categorical_crossentropy - for classification problems
2. binary_crossentropy - for binary classification problems
3. mse - for regression problems

# Metrics:
1. accuracy
2. mae, mse - for regression problems

# Techniques:
1. Early stopping is a useful technique to prevent overfitting.
2. If there is not much training data, one technique is to prefer a small network with few hidden layers to avoid overfitting.
3. Deep learning models tend to be good at fitting to the training data, but the real challenge is generalization, not fitting.


# Overfitting and Underfitting
1. Overfitting
- If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the test data. 
- how to train for an appropriate number of epochs as we'll explore below is a useful skill.
2. Underfitting
- there is still room for improvement on the test data
- model is not powerful enough, is over-regularized, or has simply not been trained long enough
  network has not learned the relevant patterns in the training data

# Prevent Overfitting
1. Get more training data (or do data-augmentation)
2. Reduce the capacity of the network.
3. Add Weight regularization 
   a. L1 Regularization - cost added is proportional to the absolute value of the weights coefficients 
   b. L2 Regularization - cost added is proportional to the square of the value of the weights coefficients; aka weight decay
4. Add Dropout
   One of the most effective and most commonly used regularization techniques for neural networks
5. Batch Normalization

# Save Models
1. Save weights only
   saving:
     cp_callback = tf.keras.callbacks.ModelCheckpoint("training_1/cp.ckpt", save_weights_only=True, verbose=1)
     model = create_model()
     model.fit(train_images, train_labels, epochs=5, batch_size=512, callbacks=[cp_callback])
   loading:
     model = create_model()
     model.load_weights(checkpoint_path)
2. Save weight values, model's configuration(architecture), optimizer configuration
   saving:
     model = create_model()
     model.fit(...)
     model.save('my_model.h5')
   loading:
     new_model = keras.models.load_model('my_model.h5')
     new_model.summary()
     loss, acc = new_model.evaluate(test_images, test_labels)

# Eager Execution
- an imperative programming environment that evaluates operations immediately, without building graphs: 
  operations return concrete values instead of constructing a computational graph to run later.
- Natural control flow (Python control flow) instead of graph control flow
- tf.enable_eager_execution()

# Tensors
- Tensors can be backed by accelerator memory (like GPU, TPU).
- Tensors are immutable
- Tensors have numpy compatibility
  TensorFlow operations automatically convert NumPy ndarrays to Tensors.
  NumPy operations automatically convert Tensors to NumPy ndarrays.

# Automatic differentiation
  
  
  
