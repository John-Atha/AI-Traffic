# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

## Project 5 - Traffic

* An AI to classify traffic signs that appear in images.
* A short presentation of the project can be found at: https://www.youtube.com/watch?v=Gw2oQew-jug

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
> > * simple2.h5 with 64 filters: `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437` (best)
> > * simple3.h5 with 128 filters: `333/333 - 7s - loss: 0.5380 - accuracy: 0.9396`
> > * simple4.h5 with 256 filters: `333/333 - 13s - loss: 0.5300 - accuracy: 0.9410`
> > * simple5.h5 with 512 filters: `333/333 - 38s - loss: 0.5634 - accuracy: 0.9412`
> > * The best number of filters seems to be 64.
> > #### Now, I will use 64 filters and try modifying the kernel size:
> > * simple2.h5 with 3x3 kernel size:  `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437` (best)
> > * simple17.h5 with 5x5 kernel size: `333/333 - 5s - loss: 0.6377 - accuracy: 0.9248`
> > * simple2.h5, with 64 filters and 3x3 kernel seems to be better
> > #### Conclusion
> > * So, the optimal convolutional filter seems to have 64 filters and 3x3 kernel
> 
> ### Pooling layer
> > #### Now, using this convolutional layer, I will try modifying the pool size of the max-pooling layer
> > * simple2.h5 with 2x2 pool size:  `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437` (best)
> > * simple20.h5 with 3x3 pool size: `333/333 - 4s - loss: 0.5462 - accuracy: 0.9326`
> > * So, increasing the pool size does not seem to lead to any progress
> > #### Conclusion
> > So, the optimal max-pooling layer seems to have 2x2 pool size.
> 
> ### Multiple convolutional and max-pooling layers
> > #### Using the previous convolutional layer with 64 filters and 3x3 kernel size and the max-pooling layer with 2x2 pool size, I will try modifying the number of these layers
> > * simple2.h5 with 1 convolutional layer:   `333/333 - 4s - loss: 0.5585 - accuracy: 0.9437`
> > * simple19.h5 with 2 convolutional layers: `333/333 - 7s - loss: 0.2389 - accuracy: 0.9583` (best)
> > #### Conclusion
> > * So, it looks like two convolutional layers lead to a better trained neural network
> 
> ### Conclusions
> > * I come to the conclusion that the optimal solution is to use two convolutional-pooling layers
> > * Each convolution layer has 64 filters with a 3x3 kernel
> > * Each pooling layer is a max-pooling layer with a 2x2 pool

## Hidden layers - Dropout
> ### Predecessor convolutional-pooling layers
> > * So far, the convolutional neural network did not have any hidden layers.
> > * I will be using the two previous optimal convolutional-pooling layers.
> > * Each convolutional layer will have 64 filters with a 3x3 kernel.
> > * Each max-pooling layer will have 2x2 pooling size.
> > * I start adding hidden layers with 'relu' activation for all of their units
>
> ### Number of hidden layer's units
> > #### At first, I am adding only one hidden layer with x units and I try modifying the x factor
> > * simple21.h5 with x=NUM_CATEGORIES:   `333/333 - 7s - loss: 3.5026 - accuracy: 0.0572`
> > * simple22.h5 with x=2*NUM_CATEGORIES: `333/333 - 6s - loss: 0.3638 - accuracy: 0.9261`
> > * simple23.h5 with x=3*NUM_CATEGORIES: `333/333 - 7s - loss: 0.2608 - accuracy: 0.9506` (best)
> > * simple24.h5 with x=4*NUM_CATEGORIES: `333/333 - 7s - loss: 0.3524 - accuracy: 0.9432`
> > * simple25.h5 with x=5*NUM_CATEGORIES: `333/333 - 7s - loss: 0.2840 - accuracy: 0.9479 `
> > #### Conclusion
> > * So, it looks like the optimal number of units is about 3*NUM_CATEGORIES
>
> ### Number of hidden layers
> > #### I will start adding more hidden layers with various units numbers, and without dropout, where x=NUM_CATEGORIES
> > * simple26.h5 with 2 hidden layers of 3x and 3x units: `333/333 - 9s - loss: 0.2708 - accuracy: 0.9405`
> > * simple47.h5 with 2 hidden layers of 3x and 1x units: `333/333 - 7s - loss: 0.1940 - accuracy: 0.9620` (best)
> > * simple28.h5 with 2 hidden layers of 3x and 2x units: `333/333 - 8s - loss: 0.2907 - accuracy: 0.9363`
> > * simple29.h5 with 2 hidden layers of 2x and 2x units: `333/333 - 7s - loss: 0.2934 - accuracy: 0.9404`
> > * simple30.h5 with 2 hidden layers of 2x and 1x units: `333/333 - 8s - loss: 0.3090 - accuracy: 0.9383`
> > * simple31.h5 with 2 hidden layers of 1x and 1x units: `333/333 - 7s - loss: 3.4992 - accuracy: 0.0504`
> > #### Conclusion 
> > * I will not try adding more hidden layers, to avoid the danger of overfitting
> > * It looks as the best combination is the model 47, with 3x and 1x units, achieving accuracy around 96%
> > * The models 26 and 29, also seem to have high accuracy
> 
> ### Dropout on one hidden layer
> > #### I will try different dropout values around 0.5 for the previous hidden layer of 3*NUM_CATEGORIES units
> > * simple23.h5 without dropout:  `333/333 - 7s - loss: 0.2608 - accuracy: 0.9506` (highest)
> > * simple37.h5 with dropout=0.2: `333/333 - 8s - loss: 0.2036 - accuracy: 0.9540` (highest)
> > * simple32.h5 with dropout=0.3: `333/333 - 7s - loss: 0.2034 - accuracy: 0.9491` (highest)
> > * simple33.h5 with dropout=0.4: `333/333 - 10s - loss: 3.4991 - accuracy: 0.0552`
> > * simple34.h5 with dropout=0.5: `333/333 - 7s - loss: 0.1820 - accuracy: 0.9474`
> > * simple35.h5 with dropout=0.6: `333/333 - 7s - loss: 0.2469 - accuracy: 0.9331`
> > * simple36.h5 with dropout=0.7: `333/333 - 8s - loss: 3.4975 - accuracy: 0.0540`
> > #### Conclusion
> > * I observe that, the larger the dropout gets, the better the testing accuracy gets compared to the training accuracy
> > * It looks like the optimal dropout for a hidden layer with 3*NUM_CATEGORIES units is at most equal to 0.3
>
> ### Dropout on multiple layers
> > #### I will try the three different dropouts 0, 0.2 and 0.3 on the layers of the models 26, 47, 29:
> >
> > #### Modifications of model 47
> > * simple42.h5 with 2 hidden layers of:
> > * * 3x units with 0.3 droppout and 1x units with 0.3 dropout:
> > * * * result: `333/333 - 8s - loss: 0.1700 - accuracy: 0.9561`
> > * simple43.h5 with 2 hidden layers of:
> > * * 3x units with 0.3 droppout and 1x units with 0.2 dropout:
> > * * * result: `333/333 - 7s - loss: 0.1483 - accuracy: 0.9616`
> > * simple44.h5 with 2 hidden layers of:
> > * * 3x units with 0.2 droppout and 1x units with 0.3 dropout:
> > * * * result: `333/333 - 7s - loss: 0.0856 - accuracy: 0.9796` (best)
> > * simple45.h5 with 2 hidden layers of:
> > * * 3x units with 0.3 droppout and 1x units with no dropout:
> > * * * result: `333/333 - 10s - loss: 0.1679 - accuracy: 0.9605`
> > * simple46.h5 with 2 hidden layers of:
> > * * 3x units with 0.2 droppout and 1x units with no dropout:
> > * * * result: `333/333 - 6s - loss: 0.2263 - accuracy: 0.9393`
> > * simple48.h5 with 2 hidden layers of:
> > * * 3x units with 0.5 droppout and 1x units with no dropout:
> > * * * result: `333/333 - 7s - loss: 0.5304 - accuracy: 0.8492`
> >
> > #### Modifications of model 26
> > * simple38.h5 with 2 hidden layers of:
> > * * 3x units with 0.3 droppout and 3x units with 0.3 dropout:
> > * * * result: `333/333 - 9s - loss: 0.1832 - accuracy: 0.9511`
> > * simple39.h5 with 2 hidden layers of:
> > * * 3x units with 0.3 droppout and 3x units with 0.2 dropout:
> > * * * result: `333/333 - 7s - loss: 0.1478 - accuracy: 0.9602`
> > * simple40.h5 with 2 hidden layers of:
> > * * 3x units with 0.2 droppout and 3x units with 0.3 dropout:
> > * * * result: `333/333 - 7s - loss: 0.1917 - accuracy: 0.9516`
> > * simple41.h5 with 2 hidden layers of:
> > * * 3x units with 0.2 droppout and 3x units with 0.2 dropout:
> > * * * result: `333/333 - 10s - loss: 0.2959 - accuracy: 0.9207`
> >
> > #### Modifications of model 29
> > * simple42.h5 with 2 hidden layers of:
> > * * 2x units with 0.3 droppout and 2x units with 0.3 dropout:
> > * * * result: `333/333 - 6s - loss: 0.2583 - accuracy: 0.9240`
> > * simple43.h5 with 2 hidden layers of:
> > * * 2x units with 0.3 droppout and 2x units with 0.2 dropout:
> > * * * result: `333/333 - 6s - loss: 0.2478 - accuracy: 0.9364`
> > * simple44.h5 with 2 hidden layers of:
> > * * 2x units with 0.2 droppout and 2x units with 0.3 dropout:
> > * * * result: `333/333 - 7s - loss: 3.5002 - accuracy: 0.0574`
> > * simple45.h5 with 2 hidden layers of:
> > * * 2x units with 0.2 droppout and 2x units with 0.2 dropout:
> > * * * result: `333/333 - 7s - loss: 0.1627 - accuracy: 0.9600`
> >
> > #### Conclusion
> > * Adding dropout seems that helps each model achieve a slightly higher accuracy around 20% than before:
> > *  * model47: 0.9620 -> max 0.9796
> > *  * model26: 0.9405 -> max 0.9602
> > *  * model29: 0.9404 -> max 0.9600

## Conclusion
> * The best model of the ones I tried seems to be the `simple44.h5` model, with 0.9796 accuracy on the testing set.
> * The code for this neural network is in the `traffic.py` file
> * This convolutional neural network consists of:
> * * One input layer
> * * Two convolutional-pooling layers, each one with:
> * * * A convolutional layer with 64 filters of a 3x3 kernel and with relu activation
> * * * A max-pooling layer of a 2x2 pool 
> * * Two hidden layers:
> * * * The first one with 3*NUM_CATEGORIES units,  relu activation
> * * * The second one with NUM_CATEGORIES units and relu activation
> * * One output layer with NUM_CATEGORIES units and softmax activation

- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com