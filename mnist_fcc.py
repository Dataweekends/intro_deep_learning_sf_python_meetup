from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Data type:", X_train.dtype)


# Exercise 1:
# Reshape your data so that each image becomes a long vector
# Your code here




# Exercise 2:
# change the type of the input vectors to be 'float32'
# and rescale them so that the values are between 0 and 1
# Your code here:



# Exercise 3:
# convert class vectors to binary class matrices
# using the keras.utils.to_categorical function
# Your code here:




# Exercise 4:
# Define a fully connected model using the Sequential API
# https://keras.io/getting-started/sequential-model-guide/
# Choose your architecture as you please
# Your code here:



# Exercise 5:
# Compile your model using an optimizer of your choice
# make sure to display the accuracy metric
# https://keras.io/optimizers/
# Your code here:




# Exercise 6:
# Fit the model on the training data. Use 30% of the
# data as validation. Experiment with different batch sizes
# and number of epochs. Save the history of training and print it.
# Your code here:




# Exercise 7:
# Calculate the score on the test data using `model.evaluate`
# Your code here:




# Bonus Exercise:
# Modify the code to use a Convolutional Neural Network
# Hints: you'll have to reshape your data to a 4D-array...