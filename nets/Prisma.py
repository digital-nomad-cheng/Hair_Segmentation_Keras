"""
Implement the portrait segmentation network by prisma
https://blog.prismalabs.ai/real-time-portrait-segmentation-on-smartphones-39c84f1b9e66
Author: yuhua cheng
Date: 11 June 2019
"""

from keras.layers import Input, Conv2D, Conv2DTranspose, DepthwiseConv2D, ReLU, Add, Activation, BatchNormalization, SeparableConv2D, UpSampling2D, Lambda, Concatenate, Multiply
from keras.activations import relu
from keras import backend as K
from keras.models import Model

def  _conv(inputs, filters, kernel, strides):
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = ReLU(max_value=6)(x)
    return x

def _conv_bn(inputs, filters, kernel, strides):
    """convolution block with batch normalization"""
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(max_value=6)(x)
    return x

def _sep_conv_bn(inputs, filters, kernel, strides, depth_activation=False):
    # depthwise
    x = DepthwiseConv2D(kernel, strides=strides, padding='same')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(max_value=6)(x)
    # pointwise
    x = Conv2D(filters, (1,1),  strides=1, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(max_value=6)(x)
    return x

def _residual_block(inputs, filters, kernel, strides, n):
    """Residual block
    """
    x = inputs
    for i in range(n):
        x0 = x
        x = SeparableConv2D(filters, kernel, strides=strides, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = ReLU(max_value=6)(x)
        X = SeparableConv2D(filters, kernel, strides=strides, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = ReLU(max_value=6)(x)
        x = Add()([x, x0])
    return x
    
def _deconv_block(inputs, filters, kernel, strides):
    x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding='same')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = ReLU(max_value=6)(x)
    return x

def PrismaNet(input_shape=(224,224,3), n_classes=1, feather=True):
    inputs = Input(shape=input_shape)
    x = _conv_bn(inputs, 16, (3,3), 1)
    x = _residual_block(x, 16, (3,3), 1, 1) 
    
    b1 = _conv_bn(x, 32, (3,3), 2)
    b1 = _residual_block(b1, 32, (3,3), 1, 1)

    b2 = _conv_bn(b1, 64, (3,3), 2)
    b2 = _residual_block(b2, 64, (3,3), 1, 1)

    b3 = _conv_bn(b2, 128, (3,3), 2)
    b3 = _residual_block(b3, 128, (3,3), 1, 2)

    b4 = _conv_bn(b3, 128, (3,3), 2)
    b4 = _residual_block(b4, 128, (3,3), 1, 6)


    up1 = UpSampling2D(size=(2,2), interpolation='bilinear')(b4)
    up1 = Add()([up1, b3])
    up1 = _conv_bn(up1, 64, (3,3), 1)
    up1 = _residual_block(up1, 64, (3,3), 1, 3, )    
    
    up2 = UpSampling2D(size=(2,2), interpolation='bilinear')(up1)
    up2 = Add()([up2, b2])
    up2 = _conv_bn(up2, 32, (3,3), 1)
    up2 = _residual_block(up2, 32, (3,3), 1, 3)

    up3 = UpSampling2D(size=(2,2), interpolation='bilinear')(up2)
    up3 = Add()([up3, b1])
    up3 = _conv_bn(up3, 16, (3,3), 1)
    up3 = _residual_block(up3, 16, (3, 3), 1, 3)

    up4 = UpSampling2D(size=(2,2), interpolation='bilinear')(up3)
    up4 = Add()([up4, x])
    up4 = _conv_bn(up4, 8, (3,3), 1)
    up4 = _residual_block(up4, 8, (3,3), 1, 3)

    up4 = Conv2D(n_classes, (1,1), padding='same', name='softmax_conv')(up4)
    output = up4 
    
    output = Activation('sigmoid')(output)

    model = Model(inputs, output)
    
    return model

if __name__ == "__main__":
    model = PrismaNet((256, 256, 3), feather=False)
    model.summary()

