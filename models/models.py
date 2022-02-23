import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
from losses import huber_pinball_loss
from tensorflow.keras.models import Model
from tensorflow_addons.losses import pinball_loss

"""
File contains the different models as well as some model util functions
"""


class wind_temp_model(tf.keras.Model):
    """
    Wind model in the format of a Keras model
    """

    def __init__(self, n_embeddings, params):
        """
        Init function for a keras model
        Args:
            n_embeddings: Number of categorical encodings for the horizon
            params: Dictionary containing optimal parameters
        """
        super(wind_temp_model, self).__init__()
        # Embedding layers
        self.embedding = Embedding(input_dim=n_embeddings, output_dim=4)
        # Create Dense layers
        self.hidden = Dense(params["n_units_1"], activation="relu")
        self.hidden2 = Dense(params["n_units_2"], activation="relu")
        self.out = Dense(5, activation="linear")
        # Create Dropout
        self.dropout = Dropout(rate=params["dropout"])

    def call(self, input_data, **kwargs):
        """
        Call function for a Keras model
        Args:
            input_data: Input data in a specified format
            **kwargs:
        Returns:
            output: Output of the whole neural network
        """
        # Extract data
        features, horizon_emb = input_data
        # Calculate embedding
        emb = self.embedding(horizon_emb)
        emb = tf.squeeze(emb, axis=1)
        conc = Concatenate(axis=1)([features, emb])
        # Calculate output
        output = self.hidden(conc)
        output = self.hidden2(output)
        output = self.dropout(output)
        output = self.out(output)
        return output


def train_wind_temp_model(train_data, train_target, validation_data, batch_size, epochs, learning_rate,
                          n_embeddings, params, label_encoder, quantiles, horizons, fine_tuning=True):
    """
    Wrapper method to train the wind/temperature model
    Args:
        train_data: Training data in specific format
        train_target: Training target
        validation_data: Validation data
        batch_size: Batch Size for training process
        epochs: Number of epochs
        learning_rate: Learning rate
        n_embeddings: Number of categorical embeddings for the horizons
        params: Optimal parameters of the respective model
        label_encoder: Label encoder for the horizons
        quantiles: List containing all quantile levels
        horizons: List containing all horizons
        fine_tuning: Boolean value, whether the model should be fine tuned on only the specific horizons

    Returns:
        model: The trained model
        [history1, history2]: List containing the history objects (can be used to access loss over the training process)

    """
    model = wind_temp_model(n_embeddings, params)
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Callbacks
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, min_delta=1e-5)
    model.compile(optimizer=optimizer, loss=lambda true, pred: huber_pinball_loss(true, pred, tau=quantiles))
    # Normal fit
    history1 = model.fit(x=train_data, y=train_target, validation_data=validation_data, epochs=epochs,
                         batch_size=batch_size, callbacks=[callback], shuffle=True, verbose=False)

    # Fine tuning
    if fine_tuning:
        enc_horizons = label_encoder.transform(horizons)
        train_filtering = np.isin(train_data[1], enc_horizons)
        train_data_fine = [train_data[0][train_filtering], train_data[1][train_filtering]]
        train_target_fine = train_target[train_filtering]
        # Val filtering
        val_data, val_target = validation_data
        val_filtering = np.isin(val_data[1], enc_horizons)
        val_data_fine = [val_data[0][val_filtering], val_data[1][val_filtering]]
        val_target_fine = val_target[val_filtering]
        validation_data_fine = (val_data_fine, val_target_fine)

        # New optimizer
        history2 = model.fit(x=train_data_fine, y=train_target_fine, validation_data=validation_data_fine,
                             epochs=epochs, batch_size=batch_size, callbacks=[callback], shuffle=True, verbose=False)
        return model, [history1, history2]
    return model, [history1]


def aggregate_wind_temp(train_data, train_target, validation_data, test_data, batch_size, epochs, learning_rate,
                        n_embeddings, params, label_encoder, quantiles, horizons, n=10):
    """
    Wrapper method to aggregate different training runs in order to average the stochastic nature of the trained network
    Args:
        train_data: Training data in specific format
        train_target: Training target
        validation_data: Validation data
        batch_size: Batch Size for training process
        epochs: Number of epochs
        learning_rate: Learning rate
        n_embeddings: Number of categorical embeddings for the horizons
        params: Optimal parameters of the respective model
        label_encoder: Label encoder for the horizons
        quantiles: List containing all quantile levels
        horizons: List containing all horizons
        n: Number of of predictions that are averaged

    Returns:
        predictions: Predictions for the test data

    """
    predictions = np.zeros(shape=(len(test_data[0]), 5))
    for i in range(n):
        model, _ = train_wind_temp_model(train_data, train_target, validation_data, batch_size, epochs, learning_rate,
                                         n_embeddings, params, label_encoder, quantiles, horizons)
        pred = model.predict(test_data)
        predictions += pred
        print("Finished Training {}".format(i + 1))
    predictions = predictions / n
    return predictions


def dax_model(window_size, dropout_rate=0.1):
    """
    Method creating and returning the dax model
    Args:
        window_size: Size of the sliding window used on the data
        dropout_rate: Dropout rate

    Returns:
        model: Keras model
    """
    # Define parameters
    filters = 16
    kernel_size = 2
    dilation_rates = [2 ** i for i in range(7)]

    # Define Inputs
    input_features = Input(shape=(window_size, 3))

    x = input_features
    # Base layers
    skips = []
    for dilation in dilation_rates:
        # Preprocessing layer
        x = Conv1D(16, 1, padding='same', activation='relu')(x)
        # Dilated convolution
        z = Conv1D(filters, kernel_size, activation="relu", padding="causal", dilation_rate=dilation)(x)
        # Postprocessing
        z = Conv1D(16, 1, padding='same', activation='relu')(z)
        # Residual connection
        x = Add()([x, z])

    # Fully Connected Layer
    out = Conv1D(32, 1, padding="same")(x)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(1, 1, activation="linear")(out)
    out = Flatten()(out)
    out = Dense(5)(out)
    model = Model(input_features, out)
    return model


def train_dax_model(model, optimizer, x_train, y_train, quantile, val_split, batch_size, epochs, callback,
                    verbose=False):
    """
    Method to train the dax model
    Args:
        model: The keras model
        optimizer: Keras opimizer
        x_train: Train features
        y_train: Train target
        quantile: Quantile to use
        val_split: Percentage of data to use for validation
        batch_size:
        epochs:
        callback:
        verbose:

    Returns:
        model: Trained model

    """
    model.compile(optimizer=optimizer, loss=lambda true, pred: pinball_loss(true, pred, tau=quantile))
    model.fit(x_train, y_train, validation_split=val_split, epochs=epochs, batch_size=batch_size, shuffle=True,
              callbacks=[callback], verbose=verbose)
    return model

def train_all_dax_models(x_train, y_train, val_split, window_size, quantiles, optimizer, batch_size, epochs, callback):
    """
    Method to train the different models for their respective quantile
    Args:
        x_train:
        y_train:
        val_split:
        window_size:
        quantiles:
        optimizer:
        batch_size:
        epochs:
        callback:

    Returns:
        models: Dictionary including a trained model for each quantile

    """
    models = dict()
    for quantile in quantiles:
        model = dax_model(window_size)
        model = train_dax_model(model, optimizer, x_train, y_train, quantile, val_split, batch_size, epochs, callback)
        print("Training finished for quantile {}".format(quantile))
        models[quantile] = model
    return models