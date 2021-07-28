import tensorflow as tf
from Utilities import model_tools, tf_data_loader


# Load our datset into train and test frames
(train_data, val_data, test_data) = tf_data_loader.load_dataset('kmnist')

# Network Design Parameters
IMAGE_SIZE = (28, 28, 1)
FILTERS = 64
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
NUM_CLASSES = 10
BATCH_SIZE = 128
DROPOUT = .4
EPOCHS = 15

# Build a CNN using our parameters and compile
kmnist_model = model_tools.define_simple_model(shape=IMAGE_SIZE,
                                               base_filters=FILTERS,
                                               kernel_size=KERNEL_SIZE,
                                               pool_size=POOL_SIZE,
                                               dropout=DROPOUT,
                                               num_classes=NUM_CLASSES)

kmnist_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                     optimizer=tf.keras.optimizers.Adam(.001),
                     metrics=['accuracy'])

# Define early stopping criteria
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')

# train our model
kmnist_model.fit(train_data,
                 validation_data=val_data,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 callbacks=callbacks)

[loss, accuracy] = kmnist_model.evaluate(test_data)
print('Test Dataset Accuracy: ', accuracy)
print('Test Dataset Loss: ', loss)