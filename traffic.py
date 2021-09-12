import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import termcolor
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43 if True else 3
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_dir [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
   
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    # Get a compiled neural network
    model = get_model()

    termcolor.cprint("--- OK TILL HERE ---", 'green')
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    termcolor.cprint("--- OK TILL HERE TOO ---", 'green')
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    def resize(img):
        return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    images = []
    labels = []
    sub_dirs = []
    
    # list all sub directories
    for _, dirs_list, _ in os.walk(os.path.join(".", data_dir)):
        sub_dirs = dirs_list
        break
    
    # traverse each sub directory and fill images - labels
    for category_dir in sub_dirs:
        for _, _, files_list in os.walk(os.path.join(".", data_dir, category_dir)):
            resized_new_images = [resize(cv2.imread(os.path.join(".", data_dir, category_dir, img))) for img in files_list]
            images += resized_new_images
            labels += [int(category_dir)] * len(files_list)
       
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # convolutional and max-pooling layers
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # convolutional and max-pooling layers
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(IMG_WIDTH/6, IMG_HEIGHT/6, 3)
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # flatten-layer
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            5*NUM_CATEGORIES, activation="relu"
        ),

        # output layer with NUM_CATEGORIES output units
        tf.keras.layers.Dense(
            NUM_CATEGORIES, activation="softmax"
        )
    ])

    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )





    return model


if __name__ == "__main__":
    main()
