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

def build_mobilenetv3_unet(input_shape):    ## (512, 512, 3)
    """ Input """
    inputs = Input(shape=input_shape)

    """ Pre-trained MobileNetV3 """
    encoder = MobileNetV3Small(include_top=False, weights="imagenet",
        input_tensor=inputs, alpha=1.0)

    """ Encoder """
    s1 = encoder.get_layer("input_1").output    # (512 x 512)
    s2 = encoder.get_layer("multiply").output   # (256 x 256)
    s3 = encoder.get_layer("re_lu_3").output    # (128 x 128)
    s4 = encoder.get_layer("multiply_1").output # (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("multiply_11").output  # (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)            # (64 x 64)
    d2 = decoder_block(d1, s3, 256)            # (128 x 128)
    d3 = decoder_block(d2, s2, 128)            # (256 x 256)
    d4 = decoder_block(d3, s1, 64)             # (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="MobileNetV3_U-Net")
    return model

MobileNetv3_model = build_mobilenetv3_unet((480, 640, 3))
# MobileNetv3_model.summary()
