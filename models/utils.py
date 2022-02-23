from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

"""
File contains util functions for the different models
"""


def normalize_wind_temp_data(dataframe, label_encoder=None, feature_scaler=None, target_scaler=None, learn=False):
    """
    Function to normalize data of the format of wind/temperature
    Args:
        dataframe: Dataframe containing the data
        label_encoder: Label encoder for the horizons
        feature_scaler: Feature scaler for normalizing the features
        target_scaler: Target scaler for normalizing the target
        learn: Boolean value to determine whether the scalers need to be learned or are given in the call

    Returns:
        data: Normalized data
        label_encoder: Label encoder with learned labels
        feature_scaler: Feature scaler with learned scaling
        target_scaler: Target scaler with learned scaling

    """
    # Drop unused columns
    data = dataframe.copy()
    data.drop(["init_tm", "met_var", "location", "ens_var", "obs_tm"], axis=1, inplace=True)
    data = data.to_numpy()
    if learn:
        label_encoder = LabelEncoder()
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        # Learn label encoding for horizons
        label = label_encoder.fit_transform(data[:, 0])
        # Learn target scaling
        target_scaled = target_scaler.fit_transform(data[:, 1].reshape(-1, 1))
        # Learn feature scaling
        features_scaled = feature_scaler.fit_transform(data[:, 2:])
        # Append
        data[:, 0] = label
        data[:, 1] = target_scaled.reshape(-1)
        data[:, 2:] = features_scaled

        return data, label_encoder, feature_scaler, target_scaler

    else:
        # Learn labels
        label = label_encoder.transform(data[:, 0])
        # Scale target
        target_scaled = target_scaler.transform(data[:, 1].reshape(-1, 1))
        # Scale features
        features_scaled = feature_scaler.transform(data[:, 2:])
        # Append
        data[:, 0] = label
        data[:, 1] = target_scaled.reshape(-1)
        data[:, 2:] = features_scaled
        return data


def convert_wind_temp_format(input_data, predict=False):
    """
    Method to convert wind/temp data to a specific format
    Args:
        input_data: Input data in a specific format
        predict: Boolean value suggesting whether the target is known or not

    Returns:
        features: All features of the data
        horizon_emb: The categorically encoded horizons
        target: The target variable

    """
    # Extract forecast embedding
    horizon_emb = input_data[:, 0]
    if not predict:
        # Extract features
        features = input_data[:, 2:]
        # Extract target
        target = np.expand_dims(input_data[:, 1], 1)
        return [features, horizon_emb], target
    else:
        # Extract features
        features = input_data[:, 1:]
        return [features, horizon_emb]

def normalize_dax_data(data, mean, sd):
    """
    Method to normalize a dataframe
    Args:
        data: Data in the format of a dataframe
        mean: Mean of a dataframe
        sd: Standard deviation of a dataframe

    Returns:
        Normalized dataframe
    """
    return (data - mean)/sd

def sliding_window(data,window_size):
    """
    Returns a sliding window view of the data
    Args:
        data: Input data
        window_size: Size of the sliding window

    Returns:
        window: Numpy array with a new dimension for the windows
    """
    window = sliding_window_view(data.to_numpy(), window_size, axis = 0)
    window = np.swapaxes(window, 1,2)
    return window

def convert_dax_data(data):
    """
    Method to convert the dax data to a specific format
    Args:
        data: Input data

    Returns:
        X: Features
        Y: Target variable
    """
    features = data[:,:,0:3]
    X = features
    Y = np.expand_dims(data[:,-1,3:],2)
    return X,Y
