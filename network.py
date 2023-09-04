# coding: utf-8
import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
from tfcnnkit.backbones.darknet_tiny import cspDarkNet53Tiny
from tfcnnkit.backbones.mobilenet_v2_tf import createMobileNetV2
from tfcnnkit.backbones.mobilenet_v3_tf import createMobileNetV3
from tfcnnkit.backbones.shufflenet_v2 import createShuffleNet
from tfcnnkit.backbones.ghostnet import createGhostNet
from tfcnnkit.backbones.efficientnet_v1 import createEfficientNetV1
from tfcnnkit.backbones.efficientnet_v2 import createEfficientNetV2
from tfcnnkit.backbones.efficientnet_lite import createEfficientNetLite
from tfcnnkit.backbones.densenet import createDenseNet
from tfcnnkit.backbones.resnet_xt import createResNetXT
from tfcnnkit.extra import activeFunction
from tfcnnkit.utils import LayerBlock


def backboneFactory(name: str,
                    input_shape: Union[list, tuple, int],
                    batch_size: int = None,
                    **kwargs) -> keras.Model:
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape, 3)
    name = name.lower()
    if "csp" in name and "dark" in name and "tiny" in name:
        return cspDarkNet53Tiny(input_shape, batch_size, **kwargs)
    if "resnet" in name:
        return createResNetXT(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "dense" in name:
        return createDenseNet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "mobilenet" in name:
        return createMobileNetV3(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "shuffle" in name:
        return createShuffleNet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "ghost" in name:
        return createGhostNet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "v1" in name:
        return createEfficientNetV1(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "v2" in name:
        return createEfficientNetV2(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "lite" in name:
        return createEfficientNetLite(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    raise NotImplementedError(name)


def createCenterNet(backbone: keras.Model,
                    input_shape: Union[list, tuple],
                    num_classes: int,
                    deconv_filters: list = [128, 128, 128],
                    head_channels: int = 64,
                    batch_size: int = None,
                    act_type: str = "swish") -> keras.Model:
    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = backbone(inputs)

    if isinstance(deconv_filters, int):
        deconv_filters = [deconv_filters] * 3

    upsamples = LayerBlock()
    for i in range(3):
        upsamples.layers.extend([
            keras.layers.Conv2DTranspose(filters=deconv_filters[i],
                                         kernel_size=4,
                                         strides=2,
                                         padding="same",
                                         kernel_initializer=keras.initializers.he_normal(),
                                         use_bias=False,
                                         name=F"deconv_{i+1}"),
            keras.layers.BatchNormalization(),
            activeFunction(act_type)
        ])
    
    x = upsamples.forward(x)

    hmap_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        activeFunction(act_type),
        tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal())
    ])
    y1 = hmap_head.forward(x)

    reg_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        activeFunction(act_type),
        tf.keras.layers.Conv2D(filters=2,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal())
    ])
    y2 = reg_head.forward(x)

    size_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        activeFunction(act_type),
        tf.keras.layers.Conv2D(filters=2,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal())
    ])
    y3 = size_head.forward(x)

    y = tf.concat([y1, y2, y3], axis=-1)

    model = keras.models.Model(inputs=inputs, outputs=y)

    return model


