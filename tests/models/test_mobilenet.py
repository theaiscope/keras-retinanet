"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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

import warnings
import pytest
import numpy as np
import keras
from keras_retinanet import losses
from keras_retinanet.models.mobilenet import allowed_backbones, mobilenet_retinanet

alphas = ['1.0']
parameters = []

for backbone in allowed_backbones:
    for alpha in alphas:
        parameters.append((backbone, alpha))


@pytest.mark.parametrize("backbone, alpha", parameters)
def test_backbone(backbone, alpha):
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    num_classes = 10

    inputs = np.zeros((1, 1024, 363, 3), dtype=np.float32)
    regression_targets = [
        np.zeros((1, 52992, 6)),
        np.zeros((1, 13248, 6)),
        np.zeros((1, 3456, 6)),
        np.zeros((1, 864, 6)),
        np.zeros((1, 216, 6)),
    ]
    classification_targets = [
        np.zeros((1, 52992, num_classes + 1)),
        np.zeros((1, 13248, num_classes + 1)),
        np.zeros((1, 3456, num_classes + 1)),
        np.zeros((1, 864, num_classes + 1)),
        np.zeros((1, 216, num_classes + 1)),
    ]
    targets = regression_targets + classification_targets

    inp = keras.layers.Input(inputs[0].shape)

    training_model = mobilenet_retinanet(num_classes=num_classes, backbone='{}_{}'.format(backbone, format(alpha)), inputs=inp)

    # compile model
    training_model.compile(
        loss={
            'regression_P3': losses.smooth_l1(),
            'regression_P4': losses.smooth_l1(),
            'regression_P5': losses.smooth_l1(),
            'regression_P6': losses.smooth_l1(),
            'regression_P7': losses.smooth_l1(),
            'classification_P3': losses.focal(),
            'classification_P4': losses.focal(),
            'classification_P5': losses.focal(),
            'classification_P6': losses.focal(),
            'classification_P7': losses.focal(),
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    training_model.fit(inputs, targets, batch_size=1)
