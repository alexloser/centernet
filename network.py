# coding: utf-8
import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
from cnnkit.backbones.darknet_tiny import csp_darknet53_tiny
from cnnkit.backbones.mobilenet_v2_tf import create_mobilenet_v2
from cnnkit.backbones.mobilenet_v3_tf import create_mobilenet_v3
from cnnkit.backbones.shufflenet_v2 import create_shufflenet
from cnnkit.backbones.ghostnet import create_ghostnet
from cnnkit.backbones.efficientnet_v1 import create_efficientnet_v1
from cnnkit.backbones.efficientnet_v2 import create_efficientnet_v2
from cnnkit.backbones.efficientnet_lite import create_efficientnet_lite
from cnnkit.backbones.densenet import create_densenet
from cnnkit.backbones.resnet_xt import create_resnet_xt
from cnnkit.extra import active_function
from cnnkit.utils import LayerBlock


def backbone_factory(name: str,
                     input_shape: Union[list, tuple, int],
                     batch_size: int = None,
                     **kwargs) -> keras.Model:
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape, 3)
    if name == "csp_darknet53_tiny":
        return csp_darknet53_tiny(input_shape, batch_size, **kwargs)
    if name == "resnet_xt":
        return create_resnet_xt(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if name == "densenet":
        return create_densenet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if name == "mobilenet":
        return create_mobilenet_v3(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if name == "shufflenet":
        return create_shufflenet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if name == "ghostnet":
        return create_ghostnet(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "v1" in name:
        return create_efficientnet_v1(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "v2" in name:
        return create_efficientnet_v2(input_shape, batch_size=batch_size, num_classes=0, **kwargs)
    if "efficient" in name and "lite" in name:
        return create_efficientnet_lite(input_shape, batch_size=batch_size, num_classes=0, **kwargs)

    raise NotImplementedError(name)


def create_centernet(backbone: keras.Model,
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
            active_function(act_type)
        ])
    
    x = upsamples.forward(x)

    hmap_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        active_function(act_type),
        tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal())
    ])
    y1 = hmap_head.forward(x)

    offset_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        active_function(act_type),
        tf.keras.layers.Conv2D(filters=2,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal())
    ])
    y2 = offset_head.forward(x)

    size_head = LayerBlock([
        tf.keras.layers.Conv2D(filters=head_channels,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               kernel_initializer=keras.initializers.he_normal()),
        active_function(act_type),
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


