import tensorflow as tf
from tensorflow.keras import layers, models
from dataclasses import dataclass
from typing import Tuple, List

from config import config

bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1


def SqueezeExcitation(input, se_ratio=1.0, name='se'):
    """
    Squeeze and Excitation block
    See more https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
    Shape: (w, h, c) -> (w, h, c)
    """
    with tf.name_scope(name):
        in_filters = input.shape[-1]
        num_reduced_filters = int(in_filters * se_ratio)

        se = layers.GlobalAveragePooling2D(name=f'{name}_squeeze')(input)
        se = layers.Reshape((1, 1, in_filters), name=f'{name}_reshape')(se)
        se = layers.Conv2D(
            filters=num_reduced_filters, kernel_size=1, activation='relu',
            padding='same', use_bias=True, name=f'{name}_reduce')(se)
        se = layers.Conv2D(
            filters=in_filters, kernel_size=1, activation='sigmoid',
            padding='same', use_bias=True, name=f'{name}_expand')(se)
        x = layers.multiply([input, se], name=f'{name}_excite')
    return x


def BottleneckBlock(input, expand_ratio=1, name='expand'):
    """
    Expands the network in depth by the given ratio.
    Also referred to as bottleneck block
    Shape: (w, h, c) -> (w, h, c * expand_ratio)
    """
    with tf.name_scope(name):
        in_filters = input.shape[-1]
        x = layers.Conv2D(
            filters=int(in_filters * expand_ratio), kernel_size=1, padding='same',
            use_bias=False, name=f'{name}_expand_conv')(input)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{name}_expand_bn')(x)
        x = layers.Activation('relu', name=f'{name}_expand_activation')(x)
    return x


def SkipConnection(input, x, dropout=0.5, name='skip'):
    """
    Skip connection. If in_filters differ from out_filters, then apply conv first on input
    Shape: (w, h, a), (w, h, b) -> (w, h, a)
    """
    with tf.name_scope(name):
        in_filters, out_filters = input.shape[3], x.shape[3]
        if x.shape[1:3] == input.shape[1:3]:
            # If input filters differ from output filter, we need to apply conv on the input image
            if in_filters != out_filters:
                input = layers.Conv2D(
                    filters=out_filters, kernel_size=1, padding='same',
                    use_bias=False, name=f'{name}_conv')(input)

            # Apply dropout on the data and sum it with input
            x = layers.Dropout(dropout, noise_shape=(None, 1, 1, 1), name=f'{name}_drop')(x)
            x = layers.add([x, input], name=f'{name}_skip')
        else:
            print('Warning: Skip connection can not be added.')
    return x


@dataclass
class HybridBlock:
    k_size: int = 3
    out_filters: int = 32
    strides: Tuple[int, int] = (1, 1)
    expand_ratio: int = 1
    se_ratio: float = 0.25
    skip_conn: bool = True


def HybridBlockLayer(input, block_args: HybridBlock, name='cblock'):
    with tf.name_scope(name):
        x = input

        # Expand in depth
        if block_args.expand_ratio > 1:
            x = BottleneckBlock(x, block_args.expand_ratio, name=f'{name}_expand')

        # Depthwise conv
        x = layers.DepthwiseConv2D(
            kernel_size=block_args.k_size, strides=block_args.strides, padding='same',
            use_bias=False, name=f'{name}_dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{name}_bn')(x)
        x = layers.Activation('relu', name=f'{name}_activation')(x)

        # Squeeze and Excitation
        if block_args.se_ratio != 1:
            x = SqueezeExcitation(x, block_args.se_ratio, name=f'{name}_se')

        # Output phase
        x = layers.Conv2D(
            filters=block_args.out_filters, kernel_size=1, padding='same',
            use_bias=False, name=f'{name}_project_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{name}_project_bn')(x)

        # Add a Skip connection. The spatial dimensions must be the same though.
        if block_args.skip_conn:
            x = SkipConnection(input, x, dropout=config.dropout, name=f'{name}_conv')

    return x


def ClassificationHead(x, n_classes, layer_units=None, name='chead_main'):
    layer_units = layer_units if layer_units is not None else []

    with tf.name_scope(name):
        # Add last layer for the "feature compression" (note - comment this if you want v1)
        x = layers.Conv2D(
            filters=x.shape[3], kernel_size=1, padding='same',
            use_bias=True, name=f'{name}_comp_conv')(x)
        x = layers.GlobalAveragePooling2D(name=f'{name}_comp_avg')(x)

        # Flattening is redundant due to previous step, but is present for formality
        x = layers.Flatten(name=f'{name}_flatten')(x)

        # Add hidden layers
        for i, units in enumerate(layer_units):
            x = layers.Dense(units, activation='relu', name=f'{name}_dense_{i}')(x)

        # Add output layer
        x = layers.Dense(n_classes, activation='softmax', dtype='float32', name=f'{name}_probs')(x)

    return x


def FramesModel(
        input_size=(224, 224, 3),
        input_filters=32,
        n_classes=10,
        blocks: List[HybridBlock] = None,
        name='frames_model',
        weights_path=None,
        freeze_blocks=0,
) -> models.Model:
    with tf.name_scope(name):
        input = layers.Input(shape=input_size, name='input')

        # Transform input
        x = layers.Conv2D(
            input_filters, 3, strides=(1, 1), padding='same',
            use_bias=False, name='input_conv')(input)
        x = layers.BatchNormalization(axis=bn_axis, name='input_bn')(x)
        x = layers.Activation(activation='relu', name='input_activation')(x)

        # Add conv blocks
        block_scopes = []
        for i, block in enumerate(blocks):
            x = HybridBlockLayer(x, block, name=f'b{i}')
            block_scopes.append(f'b{i}')

        # Add classification head if classification is specified, otherwise we return just the (spatial) feature vector
        y = ClassificationHead(x, n_classes, layer_units=[], name='chead_main') if n_classes else x

        # Conbine layers into a model
        model = models.Model(input, y, name=name)

        # Load weights
        if weights_path:
            print('Loading weights')
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        # Freeze weights
        if freeze_blocks != 0:
            print(f'Freezing weights: blocks till {freeze_blocks}')
            # Freeze input weights and a certain percentage of blocks
            freeze_scopes = ['input'] + [scope for i, scope in enumerate(block_scopes) if
                                         i < freeze_blocks or freeze_blocks < 0]
            if freeze_blocks == -1: freeze_scopes.append('comp')

            print(f'Freezing scopes: {freeze_scopes}')
            for layer in model.layers:
                if any(layer.name.startswith(scope) for scope in freeze_scopes):
                    layer.trainable = False

    return model


def prefix_model(model: models.Model, prefix: str):
    for layer in model.layers:
        layer._name = prefix + layer.name


def FusionModel(
        frames_model: models.Model,
        of_model: models.Model,
        n_classes=10,
        blocks: List[HybridBlock] = None,
        name='fusion_model',
        weights_path=None,
) -> models.Model:
    with tf.name_scope(name):
        # Freeze both models
        frames_model.trainable = False
        of_model.trainable = False
        prefix_model(frames_model, 'pim')
        prefix_model(of_model, 'pof')

        # Combine models outputs
        x = layers.concatenate([frames_model.output, of_model.output])

        # Add conv blocks
        block_scopes = []
        for i, block in enumerate(blocks):
            x = HybridBlockLayer(x, block, name=f'fb{i}')
            block_scopes.append(f'fb{i}')

        # Add a classification head
        y = ClassificationHead(x, n_classes, layer_units=[128], name='chead_main')

        # Conbine layers into a model
        model = models.Model([of_model.input, frames_model.input], y, name=name)

        # Load weights
        if weights_path:
            print('Loading weights')
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model
