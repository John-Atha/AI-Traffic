# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

### Project 5 - Traffic

* An AI to identify which traffic sign appears in a photograph.


##### Usage

`python3 traffic.py gtsrb-small`

##### Implementation

* We use [TensorFlow](https://www.tensorflow.org/) to build a `neural network` to classify road signs based on an image of them.
* We use the labeled  [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset.
* The goal was to implement the `load_data` and `get_model` functions of the `traffic.py` file.
* The `load_data` function uses the [OpenCV-Python](https://docs.opencv.org/4.5.2/index.html) module to convert images into numpy multidimensional arrays and splits the data to the `images` and `labels` lists so that the `neural network` can use them.

- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com