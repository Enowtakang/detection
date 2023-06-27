import os.path
import numpy as np
import glob
import cv2
from keras.utils import to_categorical
from sklearn import preprocessing


"""
Initial variables:
    1./ size: this defines the size which 
        the image shall take. 
        Deep learning parts are memory 
        intensive, so its better to use 
        a smaller image dimension size.

"""

size = 128


"""
Prepare images and labels:
    
    These shall be used during data loading.

"""


def prepare_images_and_labels(

        # This function takes one variable:
        # 'path'.
        # It refers to the 'path' to the
        # TRAINING or TESTING image sets.

        path):
    """
    1./ Prepare images and their labels

    """

    images = []
    labels = []

    for directory_path in glob.glob(path):

        label = directory_path.split("\\")[-1]

        # print(label)

        for image_path in glob.glob(
            os.path.join(
                directory_path, "*.*")):

            # print(image_path)

            img = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR)

            img = cv2.resize(
                img,
                (size, size))

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR)

            images.append(img)
            labels.append(label)

    return images, labels


"""
Define a powerful function to load the 
    specific type of dataset which you 
    need, from the EnowNet_Datasets
    folder.
"""


def load_dataset(

        #   This function takes in a variable
        #   called 'name', which MUST be of
        #   type 'str' (string).

        #   This variable shall be used to
        #   specify the exact dataset needed.

        #   See bottom of script for names
        #   of datasets.

        name: str):

    """
    1./ Prepare TRAINING images and their labels.
    
    """

    path_to_training_images = (
        f"C:/EnowNet_Datasets/{name}/train/*")

    training_images, training_labels = prepare_images_and_labels(
        path_to_training_images)

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)

    """
    2./ Prepare testing images and their labels
    
    """

    path_to_testing_images = (
        f"C:/EnowNet_Datasets/{name}/test/*")

    testing_images, testing_labels = prepare_images_and_labels(
        path_to_testing_images)

    testing_images = np.array(testing_images)
    testing_labels = np.array(testing_labels)

    """
    3./ Encode labels from text to integers
    
    """

    label_encoder = preprocessing.LabelEncoder()

    label_encoder.fit(training_labels)
    training_labels_encoded = label_encoder.transform(training_labels)

    label_encoder.fit(testing_labels)
    testing_labels_encoded = label_encoder.transform(testing_labels)

    """
    4./ Split data into train and test 
        datasets.
            - Not actually splitting, 
                but just performing 
                name reassignment. 
    """

    X_train, y_train, X_test, y_test = (
        training_images,
        training_labels_encoded,
        testing_images,
        testing_labels_encoded)

    """
    5./ Scale pixel values between 0 and 1
    
    """

    X_train, X_test = (
        X_train / 255.0,
        X_test / 255.0)

    """
    One-hot encode y values for neural network
    
    """

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    return (
        X_train,
        y_train,
        y_train_one_hot,

        X_test,
        y_test,
        y_test_one_hot,

        training_labels,
        testing_labels
    )


"""
NOTES:

    WHERE:
            BS - Bacterial Spot,
            LB - Late Blight,
            LM - Leaf Mold
            YLC - Yellow Leaf Curl

    The names of the datasets are:
            - bell_pepper_BS
            - tomato_LB
            - tomato_LM
            - tomato_YLC,
"""

# (
#     X_train,
#     y_train,
#     y_train_one_hot,
#
#     X_test,
#     y_test,
#     y_test_one_hot,
#
#     training_labels,
#     testing_labels
# ) = load_dataset('tomato_LB')
