# Classifiers
This project contains some image classifiers that I have put together as examples of __Machine Learning__ development work that I have done. There are currently examples of classifiers for the MNIST, Fashion MNIST, and KMNIST datasets using TensorFlow. 

1) The __KMNIST__ example uses a generic 3 layer CNN with MaxPooling after each convolutional layer. The model is created using the *define_simple_model* function, which  allows for the configuration of image size, number of filters (1st layer only, each layer scales by **2x**), kernel size, pool size, dropout rate, and the number of classes. This example uses Tensorflow Datasets, *tfds*, with a simple pipeline to both train and evaluate the networks performance.

2) The __MNIST__ example uses a deeper and more configurable CNN. The model is created using the *define_configurable_model* function, which allows for the more degrees of freedom to develop the CNN. This model has inputs for image size, number of filters (again, first layer only, each layer scales by **2x**), kernel size, pool size, dropout rate, depth, number of classes and the ability to enable or disable MaxPooling layers. This example uses tf.keras.datasets.mnist to load the data, which demonstrates a more manual input pipeline. 

3) The __Fashion MNIST__ example combines a TFDS pipeline with a model created using the *define_configurable_model* function. No additional features have been implemented.


# Areas for Improvement
Given the simplicity of these datasets, I have opted to keep the implementations of these examples straight forward. There are a number of improvements that can be made to the pipelines and the neural network architectures, such as hyperparameter tuning (either via a grid or randomized search), image preprocessing, etc. 

