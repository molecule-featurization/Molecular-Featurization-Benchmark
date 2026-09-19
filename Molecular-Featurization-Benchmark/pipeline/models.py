"""
models.py

The two network architectures used in the benchmark, and the parameter
matching that puts them on an equal footing.

Weighted Views has an architecture of its own: a base network is applied to
each view of a molecule, the resulting per-view outputs are combined by a
learned weighted sum, and a second network maps that combined vector to the
prediction.  The other four representations are single vectors and are fed
to a plain dense network.

To compare the representations rather than the capacity of the networks,
the Weighted Views model is built first, its trainable parameters are
counted, and the hidden width of the dense network is then chosen by binary
search so that it lands as close as possible to that same count.  The width
search is over integers, so the counts are matched approximately rather than
exactly; ``build_dense`` reports the count it achieved and the caller
records it.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def set_seeds(seed):
    """Seed every source of randomness that affects training."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def count_trainable_parameters(model):
    """Number of trainable parameters in a Keras model."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def dense_parameter_count(input_dim, output_dim, hidden_layers, width):
    """Parameters of a plain dense stack, without building it."""
    params = (input_dim + 1) * width
    for _ in range(hidden_layers - 1):
        params += (width + 1) * width
    params += (width + 1) * output_dim
    return params


def estimate_width(input_dim, output_dim=1, hidden_layers=3, target_params=1_000_000,
                   max_width=20_000):
    """Hidden width whose parameter count is closest to ``target_params``.

    Binary search finds the smallest width that reaches the target; the
    width one below it is the largest that stays under.  Whichever of the
    two is nearer the target is returned, so the match is not biased upward.
    """
    low, high = 1, max_width
    while low <= high:
        mid = (low + high) // 2
        params = dense_parameter_count(input_dim, output_dim, hidden_layers, mid)
        if params < target_params:
            low = mid + 1
        else:
            high = mid - 1

    upper = max(1, min(low, max_width))
    lower = max(1, upper - 1)

    upper_gap = abs(dense_parameter_count(input_dim, output_dim, hidden_layers, upper) - target_params)
    lower_gap = abs(dense_parameter_count(input_dim, output_dim, hidden_layers, lower) - target_params)

    return upper if upper_gap <= lower_gap else lower


def _final_activation(task_type):
    return "sigmoid" if task_type == "classification" else None


def _compile(model, task_type):
    if task_type == "classification":
        model.compile(optimizer="adam", loss="binary_crossentropy")
    else:
        model.compile(optimizer="adam", loss="mse")
    return model


def multi_dense(n_in, n_out, n_hidden, width=None, kernel_regularizer=None,
                name=None):
    """A stack of ``n_hidden`` ReLU layers followed by a linear output."""
    if width is None:
        width = n_in + n_out

    x = inputs = keras.Input(shape=(n_in,), name="multidense_input")
    for i in range(n_hidden):
        x = layers.Dense(
            width,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            name=f"dense{i}",
        )(x)
    outputs = layers.Dense(n_out, name="multidense_output")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_dense(input_dim, output_dim, task_type, hidden_layers=3,
                target_params=None, width=None, l2=0.01):
    """The network used for Coulomb Matrix, Bag of Bonds, ACSF and SOAP.

    Either ``width`` is given directly, or ``target_params`` is given and
    the width is chosen to match it.
    """
    if width is None:
        if target_params is None:
            raise ValueError("pass either width or target_params")
        width = estimate_width(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            target_params=target_params,
        )

    regularizer = regularizers.l2(l2) if l2 else None

    x = inputs = keras.Input(shape=(input_dim,))
    for _ in range(hidden_layers):
        x = layers.Dense(width, activation="relu", kernel_regularizer=regularizer)(x)
    outputs = layers.Dense(output_dim, activation=_final_activation(task_type))(x)

    model = _compile(keras.Model(inputs, outputs, name="dense"), task_type)

    # A representation whose input is very wide can exceed the budget even at
    # a hidden width of one, in which case the counts cannot be matched. That
    # is a property of the representation, not a bug, but it must be visible
    # rather than silently reported as a matched comparison.
    if target_params is not None and width == 1:
        achieved = dense_parameter_count(input_dim, output_dim, hidden_layers, 1)
        if achieved > target_params * 1.05:
            print(
                f"[models] input dimension {input_dim} is too large to match "
                f"{target_params:,} parameters; the narrowest network already "
                f"has {achieved:,}. The achieved count is recorded per fold."
            )

    return model, int(width)


def parallel_wrapper(n_parallel, base_model, use_max=False):
    """Apply ``base_model`` to every view and combine the per-view outputs.

    The default combination is the learned weighted sum described in the
    paper: the per-view outputs are contracted against the per-view weights
    that the featurisation supplies.  ``use_max`` replaces it with a maximum
    over views, which the paper's ablation uses.
    """
    n_in = base_model.inputs[0].shape[1]
    n_out = base_model.outputs[0].shape[1]

    parallel_inputs = keras.Input(shape=(n_parallel, n_in), name="views")
    unstacked = tf.keras.ops.unstack(parallel_inputs, n_parallel, 1)
    stacked = tf.keras.ops.stack([base_model(v) for v in unstacked], axis=1)

    weight_inputs = keras.Input(shape=(n_parallel,), name="view_weights")

    if use_max:
        out = layers.MaxPool1D(pool_size=n_parallel)(stacked)
        out = layers.Reshape((n_out,))(out)
    else:
        out = layers.Dot((-2, -1))([stacked, weight_inputs])

    return keras.Model(
        inputs=[weight_inputs, parallel_inputs],
        outputs=out,
        name="parallel_wrapper",
    )


def build_weighted_views(view_dim, n_views, output_dim, task_type,
                         base_layers=3, n_features=10, end_layers=3,
                         l2=0.01, aggregation="learned"):
    """The Weighted Views model.

    ``aggregation`` selects how the per-view predictions are combined:
    ``learned`` is the weighted sum used for the main results, and ``max``
    is the maximum-over-views variant used in the ablation.
    """
    regularizer = regularizers.l2(l2) if l2 else None

    base = multi_dense(
        view_dim,
        n_features,
        base_layers,
        kernel_regularizer=regularizer,
        name="base_network",
    )

    wrapper = parallel_wrapper(n_views, base, use_max=(aggregation == "max"))

    head = multi_dense(
        n_features,
        output_dim,
        end_layers,
        kernel_regularizer=regularizer,
        name="prediction_network",
    )

    activation = _final_activation(task_type)
    outputs = head(wrapper.outputs[0])
    if activation is not None:
        outputs = layers.Activation(activation)(outputs)

    model = keras.Model(inputs=wrapper.inputs, outputs=outputs, name="weighted_views")
    return _compile(model, task_type)


def architecture_summary(model, width=None):
    """A short record of a model's shape, for the architecture table."""
    return {
        "trainable_parameters": count_trainable_parameters(model),
        "hidden_width": width,
        "n_layers": len([l for l in model.layers if isinstance(l, layers.Dense)]),
    }
