"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from .. import initializers
from .. import layers
from .. import losses

import numpy as np

custom_objects = {
    'UpsampleLike'          : layers.UpsampleLike,
    'PriorProbability'      : initializers.PriorProbability,
    'RegressBoxes'          : layers.RegressBoxes,
    'NonMaximumSuppression' : layers.NonMaximumSuppression,
    'Anchors'               : layers.Anchors,
    '_smooth_l1'            : losses.smooth_l1(),
    '_focal'                : losses.focal(),
}


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, anchor_parameters):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]


def __build_model_pyramid(name, model, features):
    return [
        keras.layers.Lambda(
            lambda x: x,
            name='{}_P{}'.format(name, 3 + index)
        )(model(f)) for index, f in enumerate(features)
    ]


def __build_pyramid(models, features):
    return [level for pyramid in [__build_model_pyramid(n, m, features) for n, m in models] for level in pyramid]


def __build_anchors(anchor_parameters, features):
    return [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_P{}'.format(3 + i)
        )(f) for i, f in enumerate(features)
    ]


def retinanet(
    inputs,
    backbone,
    num_classes,
    anchor_parameters       = AnchorParameters.default,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    output_fpn              = False,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    # Arguments
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        anchor_parameters       : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        output_fpn              : If True, the last 5 outputs of the returned model are P3...P7.
        name                    : Name of the model.
    # Returns
        Model with inputs as input and as output the generated anchors and the output of each submodel for each pyramid level.

        The order is as defined in submodels. Using default values the output is:
        ```
        [
            anchors_P3, anchors_P4, anchors_P5, anchors_P6, anchors_P7,
            regression_P3, regression_P4, regression_P5, regression_P6, regression_P7,
            classification_P3, classification_P4, classification_P5, classification_P6, classification_P7,
        ]
        ```
    """
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)
    anchors  = __build_anchors(anchor_parameters, features)

    # concatenate the outputs to one list
    outputs = anchors + pyramids
    if output_fpn:
        outputs = features + outputs

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def retinanet_bbox(
    inputs,
    num_classes,
    nms=True,
    output_fpn=False,
    name='retinanet-bbox',
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output detections directly.

    This model uses the minimum retinanet model and appends a few layers to compute detections within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    # Arguments
        inputs      : keras.layers.Input (or list of) for the input to the model.
        num_classes : Number of classes to classify.
        output_fpn  : If True, the last 5 outputs before detections are P3...P7.
        name        : Name of the model.
        *kwargs     : Additional kwargs to pass to the minimal retinanet model.
    # Returns
        Model with inputs as input and as output the output of each submodel for each pyramid level and the detections.

        The order is as defined in submodels. Using default values the output is:
        ```
        [
            regression_P3, regression_P4, regression_P5, regression_P6, regression_P7,
            classification_P3, classification_P4, classification_P5, classification_P6, classification_P7,
            boxes_P3, boxes_P4, boxes_P5, boxes_P6, boxes_P7,
            detections
        ]
        ```
    """
    model = retinanet(inputs=inputs, num_classes=num_classes, output_fpn=output_fpn, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors        = model.outputs[:5]
    regression     = model.outputs[5:10]
    classification = model.outputs[10:15]

    if output_fpn:
        other = model.outputs[15:-5]
        fpn = model.outputs[-5:]
    else:
        other = model.outputs[15:]

    # apply predicted regression to anchors
    boxes = [
        layers.RegressBoxes(name='boxes_P{}'.format(3 + index))([a, r]) for index, (a, r) in enumerate(zip(anchors, regression))
    ]

    # concatenate all outputs to single blobs
    all_anchors        = keras.layers.Concatenate(axis=1, name='all_anchors')(anchors)
    all_regression     = keras.layers.Concatenate(axis=1, name='all_regression')(regression)
    all_classification = keras.layers.Concatenate(axis=1, name='all_classification')(classification)
    all_boxes          = keras.layers.Concatenate(axis=1, name='all_boxes')(boxes)

    # additionally apply non maximum suppression
    if nms:
        detections = layers.NonMaximumSuppression(name='nms')([all_boxes, all_classification] + other)
    else:
        detections = keras.layers.Concatenate(axis=2, name='detections')([all_boxes, classification] + other)

    # construct the list of outputs
    outputs = regression + classification + other + (fpn if output_fpn else []) + [detections]

    # construct the model
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)
