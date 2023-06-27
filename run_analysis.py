"""
STEPS:
            - Load dataset,
            - define model,
            - fit model,
            - calculate accuracy with rf classifier
"""
from load_dataset import load_dataset
from fit_model import fit_model
from random_forest_classifier import (
    random_forest_classifier)

from models import (
    le_net_5_model,
    alex_net_model,
    zf_net_model,
    vgg_net_model
)

from detection import detection_model


"""
Load dataset
"""
(
    X_train,
    y_train,
    y_train_one_hot,

    X_test,
    y_test,
    y_test_one_hot,

    training_labels,
    testing_labels
    ) = load_dataset('bell_pepper_BS')


"""
Define model
"""
leNet_5 = le_net_5_model()
alex_net = alex_net_model()
zf_net = zf_net_model()
vgg_net = vgg_net_model()
detection = detection_model()

"""
Fit model
"""
nn_model = fit_model(

    # Here, name the already-loaded model
    detection,

    # Leave these untouched
    X_train,
    X_test,
    y_train_one_hot,
    y_test_one_hot)


"""
Calculate accuracy with rf classifier
"""
accuracy = random_forest_classifier(
    nn_model,
    X_train,
    y_train,
    X_test,
    testing_labels)

print(accuracy)
