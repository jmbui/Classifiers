import numpy as np
import tensorflow as tf
from Utilities import model_tools


def normalize_image_data(dataset_images):
    """
    This function is used to normalize the images to values [0-1]
    :param dataset_images: Images to be normalized
    :return: The normalized image data
    """
    dataset_images = dataset_images / 255.0
    return dataset_images


def one_hot_classes(dataset_labels):
    """
    This function is used to one-hot encode the labels
    :param dataset_labels: Labels to be encoded
    :return: The encoded labels
    """
    return tf.keras.utils.to_categorical(dataset_labels)


def split_train_val(training_images, training_labels):
    """
    This function parses the last 10000 items and places them into a validation dataset
    :param training_images: Image dataset to be split
    :param training_labels: Label dataset to be split
    :return: The training and validation image/label datasets
    """
    train_images = training_images[:-10000]
    valid_images = training_images[-10000:]
    train_labels = training_labels[:-10000]
    valid_labels = training_labels[-10000:]

    return train_images, valid_images, train_labels, valid_labels


# Load the MNIST dataset, normalize the images, and one hot encode classes
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = normalize_image_data(train_x)
test_x = normalize_image_data(test_x)
train_y = one_hot_classes(train_y)
test_y = one_hot_classes(test_y)

# The images need to be reshaped to include the channel information
train_x = np.reshape(train_x, [-1, 28, 28, 1])
test_x = np.reshape(test_x, [-1, 28, 28, 1])

# Grab 10000 images/labels for validation
train_x, valid_x, train_y, valid_y = split_train_val(train_x, train_y)

# Network Design Parameters
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
FILTERS = 32
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT = .25
DEPTH = 4
EPOCHS = 15
BATCH_SIZE = 64


# Define the model using the parameters above
no_pooling_mnist_model = model_tools.define_configurable_model(shape=IMG_SHAPE,
                                                               base_filters=FILTERS,
                                                               kernel_size=KERNEL_SIZE,
                                                               pool_size=POOL_SIZE,
                                                               dropout=DROPOUT,
                                                               num_classes=NUM_CLASSES,
                                                               depth=DEPTH,
                                                               with_pooling=False)

# Compile the model
no_pooling_mnist_model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=tf.keras.optimizers.Adam(.001),
                               metrics=['accuracy'])

# Define early stopping criteria
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')

# Train and validate our model
no_pooling_mnist_model.fit(train_x, train_y,
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           validation_data=(valid_x, valid_y),
                           callbacks=callbacks)

# Evaluate the model using the test dataset, print the loss and accuracy data
[loss, accuracy] = no_pooling_mnist_model.evaluate(test_x, test_y)
print('Test Dataset Accuracy: ', accuracy)
print('Test Dataset Loss: ', loss)
