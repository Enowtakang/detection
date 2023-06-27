"""
This classifier takes in features which
    have been extracted by a neural network,
    and makes a BINARY classification.

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing


"""
Initial Variables

"""

n_estimators = 100
random_state = 2023


"""
Define a function to implement this classifier.
    
    This function takes in a NEURAl NETWORK
        model as input, amongst OTHERS.

"""


def random_forest_classifier(
        nn_model,
        X_train, y_train,
        X_test,
        testing_labels):

    """
    Extract 'training' features from
        the neural network

    """

    training_features = nn_model.predict(X_train)

    """
    Instantiate Random Forest Classifier
    
    """

    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state)

    """
    Train the classifier on training data
    """

    rf_classifier.fit(training_features, y_train)

    """
    Extract 'testing' features from 
        the neural network
    
    """

    testing_features = nn_model.predict(X_test)

    """
    Make predictions using the classifier
    
    """

    predictions = rf_classifier.predict(
        testing_features)

    # """
    # Inverse label encoder transform to get
    # original label back.
    #
    # """
    #
    # label_encoder = preprocessing.LabelEncoder()
    #
    # predictions = label_encoder.inverse_transform(
    #     predictions)

    """
    Print overall accuracy
    
    """

    accuracy = metrics.accuracy_score(
        testing_labels,
        predictions)

    return accuracy


# accuracy = random_forest_classifier(
#     nn_model,
#     X_train,
#     y_train,
#     X_test,
#     testing_labels)
#
# print(accuracy)
