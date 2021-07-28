import tensorflow as tf
from Utilities import model_tools, tf_data_loader

# Use the TFDS pipeline from the tf_data_loader script
(train_data, val_data, test_data) = tf_data_loader.load_dataset('fashion_mnist')

# Network Design Parameters
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
FILTERS = 32
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT = .4
DEPTH = 4
EPOCHS = 15
BATCH_SIZE = 64

# Define our more customized model using the above parameters
fashion_model = model_tools.define_configurable_model(shape=IMG_SHAPE,
                                                      base_filters=FILTERS,
                                                      kernel_size=KERNEL_SIZE,
                                                      pool_size=POOL_SIZE,
                                                      dropout=DROPOUT,
                                                      num_classes=NUM_CLASSES,
                                                      depth=DEPTH,
                                                      with_pooling=False)


# Compile the model
fashion_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(.001),
                      metrics=['accuracy'])

# Define early stopping criteria
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto')

# Train and validate our model
fashion_model.fit(train_data,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=val_data,
                  callbacks=callbacks)

# Evaluate the model using the test dataset, print the loss and accuracy data
[loss, accuracy] = fashion_model.evaluate(test_data)
print('Test Dataset Accuracy: ', accuracy)
print('Test Dataset Loss: ', loss)
