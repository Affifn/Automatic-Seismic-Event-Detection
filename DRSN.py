from __future__ import print_function
import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout, Flatten
from keras.layers import AveragePooling1D, Input, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
K.set_learning_phase(1)


def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return inputs


def sign_backend(inputs):  #
    return K.sign(inputs)


def sign_exp(inputs):
    return K.exp(inputs)



def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = K.expand_dims(inputs, -1)
    inputs = K.spatial_2d_padding(inputs, ((0, 0), (pad_dim, pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)


# Residual Shrinakge Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    for i in range(nb_blocks):

        identity = residual
        in_channels = identity.shape[-1]
        if not downsample:
            downsample_strides = 1

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, strides=(downsample_strides),
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)

        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        abs_mean_ = GlobalAveragePooling1D()(residual)
        scales_ = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(abs_mean_)
        scales_ = BatchNormalization()(scales_)
        scales_ = Activation('relu')(scales_)
        scales_ = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales_)
        scales_ = Lambda(expand_dim_backend)(scales_)

        thres = keras.layers.multiply([abs_mean, scales])


        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        residual = keras.layers.multiply([residual, scales_])

        if downsample_strides > 1:
            identity = AveragePooling1D(pool_size=1, strides=2)(identity)

        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)

        residual = keras.layers.add([residual, identity])

    return residual