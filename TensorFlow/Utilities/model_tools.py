import tensorflow as tf
import tensorflow.keras.layers as layers


def define_simple_model(shape, base_filters=8, kernel_size=(3, 3), pool_size=(2, 2),
                        dropout=.25, num_classes=1):
    """
    This function creates a simple Convolutional Neural Network for image classification

    :param shape: Input Image Shape
    :param base_filters: The base number of filters to be used in the convolution layers
    :param kernel_size: kernel size of the convolutions
    :param pool_size: MaxPooling pool size
    :param dropout: Dropout settings for training
    :param num_classes: Number of Classes in the dataset
    :return: Returns a CNN model based on the input parameters
    """

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(input_shape=shape, filters=base_filters, kernel_size=kernel_size,
                            activation=tf.keras.activations.relu))
    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.Conv2D(filters=base_filters * 2, kernel_size=kernel_size,
                            activation=tf.keras.activations.relu))
    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.Conv2D(filters=base_filters * 4, kernel_size=kernel_size,
                            activation=tf.keras.activations.relu))
    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation(tf.keras.activations.softmax))

    return model


def define_configurable_model(shape, base_filters=8, kernel_size=(3, 3), pool_size=(2, 2),
                              dropout=.25, num_classes=1, depth=1, with_pooling=True):
    """
    This function creates a configurable Convolutional Neural Network for image classification


    :param shape: Input Image Shape
    :param base_filters: The base number of filters to be used in the convolution layers
    :param kernel_size: kernel size of the convolutions
    :param pool_size: MaxPooling pool size
    :param dropout: Dropout settings for training
    :param num_classes: Number of Classes in the dataset
    :param depth: How many 2D Convolution layers are used
    :param with_pooling: Whether Pooling should be used after each convolution layer
    :return: Returns a CNN model based on the input parameters
    """

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(input_shape=shape, filters=base_filters, kernel_size=kernel_size,
                            activation=tf.keras.activations.relu))
    if with_pooling:
        model.add(layers.MaxPool2D(pool_size=pool_size))

    # Create deeper layers by iterating through depth
    if depth > 1:
        for i in range(depth):
            model.add(layers.Conv2D(filters=base_filters * 2, kernel_size=kernel_size,
                                    activation=tf.keras.activations.relu))
            if with_pooling:
                model.add(layers.MaxPool2D(pool_size=pool_size))

    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation(tf.keras.activations.softmax))

    return model
