from tensorflow_addons.utils.types import TensorLike, FloatTensorLike
from tensorflow_addons.losses import pinball_loss
import tensorflow as tf

"""
File includes all different losses I used in my models and experiments
"""

#Smooth exp quantile loss
@tf.function
def exp_pinball_loss(
    y_true: TensorLike, y_pred: TensorLike, tau: FloatTensorLike = 0.5,
    alpha: FloatTensorLike = 0.001
) -> tf.Tensor:
    """Computes the exp-pinball loss between `y_true` and `y_pred`.
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
        tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
            shape = `[d0,..., dn]`.
        alpha: (Optional) Float > 0. Smoothing parameter for the loss function.

    Returns:
        loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Broadcast the pinball slope along the batch dimension
    tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
    alpha = tf.expand_dims(tf.cast(alpha, y_pred.dtype), 0)

    delta_y = y_true - y_pred
    #Implement smooth loss
    pinball = tau * delta_y + alpha * tf.math.softplus(-delta_y/alpha)
    return tf.reduce_mean(pinball, axis=-1)#


#Smooth sqrt quantile loss
@tf.function
def sqrt_pinball_loss(
    y_true: TensorLike, y_pred: TensorLike, tau: FloatTensorLike = 0.5,
    alpha: FloatTensorLike = 0.001
) -> tf.Tensor:
    """Computes sqrt-the pinball loss between `y_true` and `y_pred`.
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
        tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
            shape = `[d0,..., dn]`.
        alpha: (Optional) Float > 0. Smoothing parameter for the loss function.
    Returns:
        loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Broadcast the pinball slope along the batch dimension
    tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
    alpha = tf.expand_dims(tf.cast(alpha, y_pred.dtype), 0)
    one = tf.cast(1, tau.dtype)

    delta_y = y_true - y_pred
    #Implement smooth loss
    pinball = (delta_y*(2*tau - one) + tf.math.sqrt(tf.math.square(delta_y) + alpha))/2
    return tf.reduce_mean(pinball, axis=-1)


# Huber quantile loss
@tf.function
def huber_pinball_loss(
        y_true: TensorLike, y_pred: TensorLike, tau: FloatTensorLike = 0.5,
        alpha: FloatTensorLike = 0.001
) -> tf.Tensor:
    """Computes Huber the pinball loss between `y_true` and `y_pred`.
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
        tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
            shape = `[d0,..., dn]`.
        alpha: (Optional) Float > 0. Smoothing parameter for the loss function.
    Returns:
        loss: 1-D float `Tensor` with shape [batch_size].
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Broadcast the pinball slope along the batch dimension
    tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
    alpha = tf.expand_dims(tf.cast(alpha, y_pred.dtype), 0)
    one = tf.cast(1, tau.dtype)

    error = tf.subtract(y_true, y_pred)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
    huber = tf.where(abs_error <= alpha, half * tf.square(error) / alpha,
                     abs_error - half * alpha)

    # Implement smooth loss
    pinball = tf.where(error >= 0, tau * huber, (one - tau) * huber)
    return tf.reduce_mean(pinball, axis=-1)