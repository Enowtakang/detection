"""
After defining the model architecture
    (at Flatten()),
    the model needs to be fit

"""
from keras.models import Model
from keras.layers import Dense


"""
Initial variables

"""
number_of_neurons = 256
activation = 'sigmoid'
kernel_initializer = 'he_uniform'

optimizer = 'rmsprop'
loss = 'categorical_crossentropy'

epochs = 1


"""
Define the fit_model function

"""


def fit_model(
        model,
        X_train,
        X_test,
        y_train_one_hot,
        y_test_one_hot):

    x = model.output

    x = Dense(
        number_of_neurons,
        activation=activation,
        kernel_initializer=kernel_initializer
    )(x)

    """
    Prediction layer 
    
    """

    prediction_layer = Dense(
        2,
        # should use sigmoid for
        # binary classification
        activation=activation
    )(x)

    """
    Make a new model which combines both 
        the 'model' and 'x'
    
    """

    nn_model = Model(
        inputs=model.input,
        outputs=prediction_layer
    )

    """
    Compile the new model
    
    """

    nn_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    """
    Train the new model
    
    """

    nn_model.fit(
        X_train, y_train_one_hot,
        epochs=epochs,
        validation_data=(X_test, y_test_one_hot)
    )

    return nn_model


# nn_model = fit_model(
#     model,
#     X_train,
#     X_test,
#     y_train_one_hot,
#     y_test_one_hot)
