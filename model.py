import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small
tf.keras.backend.clear_session()

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_mobilenetv3_unet(input_shape):    ## (480, 640, 3)
    """ Input """
    inputs = Input(shape=input_shape)

    """ Pre-trained MobileNetV3 """
    encoder = MobileNetV3Small(include_top=False, weights=None,
        input_tensor=inputs)

    """ Encoder """
    e1 = encoder.get_layer("input_1").output    # (480 x 640)
    e2 = encoder.get_layer("multiply").output   # (240 x 320)
    e3 = encoder.get_layer("re_lu_3").output    # (120 x 160)
    e4 = encoder.get_layer("multiply_1").output # (60 x 80)

    """ Bridge """
    b1 = encoder.get_layer("multiply_11").output  # (30 x 40)

    """ Decoder """
    d1 = decoder_block(b1, e4, 512)            # (60 x 80)
    d2 = decoder_block(d1, e3, 256)            # (120 x 160)
    d3 = decoder_block(d2, e2, 128)            # (240 x 320)
    d4 = decoder_block(d3, e1, 64)             # (480 x 640)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="MobileNetV3_U-Net")
    return model

MobileNetv3_model = build_mobilenetv3_unet((480, 640, 3))
# MobileNetv3_model.summary()
