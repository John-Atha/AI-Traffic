# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

## Project 5 - Traffic

* An AI to identify which traffic sign appears in a photograph.


### Usage

`python3 traffic.py gtsrb-small`

### Implementation

* We use [TensorFlow](https://www.tensorflow.org/) to build a `neural network` to classify road signs based on an image of them.
* We use the labeled  [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset.
* The goal was to implement the `load_data` and `get_model` functions of the `traffic.py` file.
* The `load_data` function uses the [OpenCV-Python](https://docs.opencv.org/4.5.2/index.html) module to convert images into numpy multidimensional arrays and splits the data to the `images` and `labels` lists so that the `neural network` can use them.

### Experimentation Proccess

* At first, I begun testing with a very simple convolutional neural network
* As I moved on, I started to use more complex neural networks

#### simple1.h5
* A simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * no hidden layers
    * RESULT: `333/333 - 3s - loss: 0.7034 - accuracy: 0.9353`
```python
    model = tf.keras.models.Sequential([
        # convolution layer
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # max-pooling with 2x2 pool
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),
        # flatten-layer
        tf.keras.layers.Flatten(),
        # no hidden layers
        # output layer with NUM_CATEGORIES output units
        tf.keras.layers.Dense(
            NUM_CATEGORIES, activation="softmax"
        )
    ])
```

#### simple2.h5
* A simple convolutional network, with:
    * one convolutional layer with 64 filters `double than before` using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * no hidden layers
    * RESULT: `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`

#### simple3.h5
* A simple convolutional network, with:
    * one convolutional layer with 128 filters `double than before` using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * no hidden layers
    * RESULT: `333/333 - 7s - loss: 0.5380 - accuracy: 0.9396`

#### simple4.h5
* A simple convolutional network, with:
    * one convolutional layer with 256 filters `double than before` using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * no hidden layers
    * RESULT: `333/333 - 13s - loss: 0.5300 - accuracy: 0.9510`

#### simple5.h5
* A simple convolutional network, with:
    * one convolutional layer with 512 filters `double than before` using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * no hidden layers
    * RESULT: `333/333 - 38s - loss: 0.5634 - accuracy: 0.9412`


- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com