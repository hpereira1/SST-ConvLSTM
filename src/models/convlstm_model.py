import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, Conv2D, BatchNormalization,
    SpatialDropout2D, Activation, Concatenate, Layer
)
from typing import Tuple, Optional


class LastFrameSlice(Layer):

    def call(self, x):
        return x[:, -1, :, :, :]

    def get_config(self):
        return super().get_config()


def build_convlstm_variant(
    input_shape: Tuple[int, int, int, int],
    n_horizons: int,
    variant: str,
    base_filters: int = 32,
    kernel_size: Tuple[int, int] = (3, 3),
    dropout: float = 0.1,
    recurrent_dropout: float = 0.1,
    spatial_dropout: float = 0.2,
    use_bn: bool = True,
    n_layers: int = 2,
) -> Model:
    """
    Construtor unificado de ConvLSTM. A arquitetura é uma pilha de `n_layers`
    camadas ConvLSTM2D (BN opcional após cada uma; dropout, recurrent_dropout
    e spatial_dropout opcionais), e as variantes diferem em dois interruptores
    derivados de `variant`:

    multihead: cada horizonte recebe sua própria cabeça (Conv2D 3x3 + ReLU +
        Conv2D 1x1); quando False, uma única Conv2D 1x1 produz todos os
        horizontes.
    skip: a saída é `delta + último frame observado` (previsão residual);
        quando False, a saída é a previsão direta.

    `variant` deve ser uma das chaves de `variant_specs`. A aplicação de BN, o
    valor dos dropouts e o número de camadas são controlados por `use_bn`, pelos
    argumentos de dropout e por `n_layers` (definidos pelo caller, train2.py).
    """
    variant_specs = {
        'vbase':         dict(multihead=False, skip=False),
        'vbase2camadas': dict(multihead=False, skip=False),
        'v0':       dict(multihead=False, skip=False),
        'v3':       dict(multihead=True,  skip=False),
        'v4':       dict(multihead=False, skip=True),
        'v5':       dict(multihead=True,  skip=True),
        'v0_clean': dict(multihead=False, skip=False),
        'v3_clean': dict(multihead=True,  skip=False),
        'v4_clean': dict(multihead=False, skip=True),
        'v5_clean': dict(multihead=True,  skip=True),
        'v4_nobn':  dict(multihead=False, skip=True),
        'v5_nobn':  dict(multihead=True,  skip=True),
        'v3_clean_nobn': dict(multihead=True,  skip=False),
        'v4_clean_nobn': dict(multihead=False, skip=True),
        'v5_clean_nobn': dict(multihead=True,  skip=True),
    }
    if variant not in variant_specs:
        raise ValueError(
            f"variant must be one of {list(variant_specs)}, got '{variant}'"
        )
    multihead = variant_specs[variant]['multihead']
    skip      = variant_specs[variant]['skip']

    inp = Input(shape=input_shape, name='sst_input')

    if skip:
        last_frame = LastFrameSlice(name='last_frame')(inp)

    x = inp
    for i in range(n_layers):
        is_last = (i == n_layers - 1)
        x = ConvLSTM2D(
            base_filters, kernel_size, padding='same',
            return_sequences=not is_last,
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            name=f'convlstm_{i+1}',
        )(x)
        if use_bn:
            x = BatchNormalization(name=f'bn_{i+1}')(x)
    if spatial_dropout > 0:
        x = SpatialDropout2D(spatial_dropout, name='spatial_drop')(x)

    if multihead:
        heads = []
        for i in range(n_horizons):
            h = Conv2D(base_filters, 3, padding='same', name=f'head_{i}_conv1')(x)
            h = Activation('relu', name=f'head_{i}_relu')(h)
            h = Conv2D(
                1, 1, padding='same', activation=None,
                dtype=tf.float32, name=f'head_{i}_out',
            )(h)
            heads.append(h)
        out_pre = heads[0] if n_horizons == 1 else \
            Concatenate(axis=-1, name='concat_heads')(heads)
    else:
        out_pre = Conv2D(
            n_horizons, 1, padding='same', activation=None,
            dtype=tf.float32, name='output',
        )(x)

    if skip:
        out = tf.keras.layers.Add(name='residual_add')([out_pre, last_frame])
    else:
        out = out_pre

    model = Model(inputs=inp, outputs=out, name=f'ConvLSTM_SST_{variant}')
    return model
