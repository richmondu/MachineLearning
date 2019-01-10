# TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: 
# operations return concrete values instead of constructing a computational graph to run later.

from __future__ import absolute_import, division, print_function

import tensorflow as tf

# Enabling eager execution changes how TensorFlow operations behave
# —now they immediately evaluate and return their values to Python.
# Now you can run TensorFlow operations and the results will return immediately:
tf.enable_eager_execution()
print("Executing eagerly: " + str(tf.executing_eagerly()))


####################################################################################
# Tensors
# A Tensor is a multi-dimensional array. Similar to NumPy ndarray objects, Tensor objects have a data type and a shape. 
# The most obvious differences between NumPy arrays and TensorFlow Tensors are:
#   Tensors can be backed by accelerator memory (like GPU, TPU).
#   Tensors are immutable
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)


####################################################################################
# NumPy Compatibility
# Conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:
#   TensorFlow operations automatically convert NumPy ndarrays to Tensors.
#   NumPy operations automatically convert Tensors to NumPy ndarrays.
import numpy as np

ndarray = np.ones([3, 3])
print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


####################################################################################
# GPU acceleration
# Many TensorFlow operations can be accelerated by using the GPU for computation. 
# Without any annotations, TensorFlow automatically decides whether to use the GPU or CPU for an operation (and copies the tensor between CPU and GPU memory if necessary).

x = tf.random_uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000])
  assert x.device.endswith("CPU:0")

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")


####################################################################################
# Datasets
# You can use Python iteration over the tf.data.Dataset object and do not need to explicitly create an tf.data.Iterator object. 

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)

# Apply transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)
print('\nElements in ds_file:')
for x in ds_file:
  print(x)