# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

## Project 5 - Traffic

* An AI to classify traffic signs that appear in photographs.


## Usage

`python3 traffic.py gtsrb-small`

## Implementation

* We use [TensorFlow](https://www.tensorflow.org/) to build a `neural network` to classify road signs based on an image of them.
* We use the labeled  [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset.
* The goal was to implement the `load_data` and `get_model` functions of the `traffic.py` file.
* The `load_data` function uses the [OpenCV-Python](https://docs.opencv.org/4.5.2/index.html) module to convert images into numpy multidimensional arrays and splits the data to the `images` and `labels` lists so that the `neural network` can use them.

## Experimentation Proccess

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
## In the following models, I will use the same NN and modify certain characteristics each time

## Convolutional-pooling layers
> ### Convolutional layer
> > #### I modify the number of the filters of the convolutional layer:
> > * simple2.h5 with 64 filters: `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`
> > * simple3.h5 with 128 filters: `333/333 - 7s - loss: 0.5380 - accuracy: 0.9396`
> > * simple4.h5 with 256 filters: `333/333 - 13s - loss: 0.5300 - accuracy: 0.9510`
> > * simple5.h5 with 512 filters: `333/333 - 38s - loss: 0.5634 - accuracy: 0.9412`
> > * The best number of filters seems to be 64.
> > #### Now, I will use 64 filters and try modifying the kernel size:
> > * simple2.h5 with 3x3 kernel size:  `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`
> > * simple17.h5 with 5x5 kernel size: `333/333 - 5s - loss: 0.6377 - accuracy: 0.9248`
> > * simple2.h5, with 64 filters and 3x3 kernel seems to be better
> > #### Conclusion
> > * So, the optimal convolutional filter seems to have 64 filters and 3x3 kernel
> 
> ### Pooling layer
> > #### Now, using this convolutional layer, I will try modifying the pool size of the max-pooling layer
> > * simple2.h5 with 2x2 pool size:  `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`
> > * simple20.h5 with 3x3 pool size: `333/333 - 4s - loss: 0.5462 - accuracy: 0.9326`
> > * So, increasing the pool size does not seem to lead to any progress
> > #### Conclusion
> > So, the optimal max-pooling layer seems to have 2x2 pool size.
> 
> ### Multiple convolutional and max-pooling layers
> > #### Using the previous convolutional layer with 64 filters and 3x3 kernel size and the max-pooling layer with 2x2 pool size, I will try modifying the number of these layers
> > * simple2.h5 with 1 convolutional layer:   `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`
> > * simple19.h5 with 2 convolutional layers: `333/333 - 7s - loss: 0.2389 - accuracy: 0.9583`
> > #### Conclusion
> > * So, it looks like two convolutional layers lead to a better trained neural network
> 
> ### Conclusions
> > * I come to the conclusion that the optimal solution is to use two convolutional-pooling layers
> > * Each convolution layer has 64 filters with a 3x3 kernel
> > * Each pooling layer is a max-pooling layer with a 2x2 pool

## Hidden layers
> ### Predecessor concolutional-pooling layers
> > * So far, the convolutional neural network did not have any hidden layers.
> > * I will be using the two previous optimal convolutional-pooling layers.
> > * Each convolutional layer will have 64 filters with a 3x3 kernel.
> > * Each max-pooling layer will have 2x2 pooling size.
> > * I start adding hidden layers with 'relu' activation for all of their units
> ### Number of hidden layer's units
> > #### At first, I am adding only one hidden layer with x units and i try modifying the x factor
> > * simple21.h5 with x=NUM_CATEGORIES:   `333/333 - 7s - loss: 3.5026 - accuracy: 0.0572`
> > * simple22.h5 with x=2*NUM_CATEGORIES: `333/333 - 6s - loss: 0.3638 - accuracy: 0.9261`
> > * simple23.h5 with x=3*NUM_CATEGORIES: `333/333 - 7s - loss: 0.2514 - accuracy: 0.9544`
> > * simple24.h5 with x=4*NUM_CATEGORIES: `333/333 - 7s - loss: 0.3524 - accuracy: 0.9432`
> > * simple25.h5 with x=5*NUM_CATEGORIES: `333/333 - 7s - loss: 0.2840 - accuracy: 0.9479 `
> > #### Conclusion
> > * So, it looks like the optimal number of units is about 3*NUM_CATEGORIES
> ### Number of hidden layers
> > #### I will start adding more hidden layers, at first with x={3*NUM_CATEGORIES} units at each one
> > * simple26.h5 with 2 hidden layers of x units each: 
> > 
> > 
> > 
- - - 
#### simple6.h5
* Now I am `adding a hidden layer` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with two units and relu activation
    * RESULT: `333/333 - 2s - loss: 3.5001 - accuracy: 0.0545`

#### simple7.h5
* Now I am `adding one more unit to the hidden layer` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with three units and relu activation
    * RESULT: `333/333 - 2s - loss: 3.4926 - accuracy: 0.0556`

#### simple8.h5
* Now I am `increase the hidden layer's units` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with `NUM_CATEGORIES units` and relu activation
    * RESULT: `333/333 - 3s - loss: 3.4961 - accuracy: 0.0568`

#### simple9.h5
* Now I am `increase even more the hidden layer's units` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with `2*NUM_CATEGORIES units` and relu activation
    * RESULT: `333/333 - 4s - loss: 0.5583 - accuracy: 0.9252`

#### simple10.h5
* Now I am `trying to find the hidden layer's units border` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with `NUM_CATEGORIES+1 units` and relu activation
    * RESULT: `333/333 - 3s - loss: 3.4940 - accuracy: 0.0517`

#### simple11.h5
* Now I am `increase even more the hidden layer's units` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with `3*NUM_CATEGORIES units` and relu activation
    * RESULT: `333/333 - 3s - loss: 0.5007 - accuracy: 0.9384`

#### simple12.h5
* Now I am `increase the hidden layers` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * two hidden layers with `2*NUM_CATEGORIES units` each and relu activation
    * RESULT: `333/333 - 3s - loss: 0.4365 - accuracy: 0.9265`

 #### simple13.h5
* Now I am `increase the hidden layers' units` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * two hidden layers with `3*NUM_CATEGORIES units` each and relu activation
    * RESULT: `333/333 - 3s - loss: 0.4693 - accuracy: 0.9241`

 #### simple15.h5
* Now I am `increase the hidden layers` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * three hidden layers with `2*NUM_CATEGORIES units` each and relu activation
    * RESULT: `333/333 - 3s - loss: 0.5186 - accuracy: 0.8684`

 #### simple14.h5
* Now I am `increase the hidden layers' units` and I have a simple convolutional network, with:
    * one convolutional layer with 32 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * three hidden layers with `3*NUM_CATEGORIES units` each and relu activation
    * RESULT: `333/333 - 3s - loss: 0.3721 - accuracy: 0.9291`

#### simple16.h5
* Now I  I have a simple convolutional network, with:
    * one convolutional layer with 64 filters using a 3x3 kernel
    * one max-pooling layer using 2x2 pool size
    * one hidden layer with `2*NUM_CATEGORIES units` each and relu activation
    * RESULT: `333/333 - 5s - loss: 0.5114 - accuracy: 0.9275`



- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com