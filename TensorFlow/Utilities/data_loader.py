import tensorflow_datasets as tfds
import tensorflow as tf


def normalize_image(image, label):
    """
    Function to normalize input image to values [0-1]
    :param image: input image
    :param label: label of the input image
    :return: normalized image and the original label
    """

    return tf.cast(image, tf.float32) / 255.0, label


def load_dataset(dataset):
    """
    Wrapper function to load and normalize tensorflow datasets
    :param dataset: The name of the dataset to be used
    :return: The train, validation, and test datasets
    """

    (train_data, val_data, test_data), dataset_info = tfds.load(dataset,
                                                      split=['train[:80%]',
                                                             'train[80%:]',
                                                             'test'],
                                                      shuffle_files=True,
                                                      as_supervised=True,
                                                      with_info=True)

    # Build the train_data input pipeline
    train_data = train_data.map(normalize_image,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.cache()
    train_data = train_data.shuffle(dataset_info.splits['train'].num_examples)
    train_data = train_data.batch(128)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    # Build the val_data input pipeline
    val_data = val_data.map(normalize_image,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(128)
    val_data = val_data.cache()
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)

    # Build the test_data input pipeline
    test_data = test_data.map(normalize_image,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(128)
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, val_data, test_data
