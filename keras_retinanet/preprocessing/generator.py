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

import numpy as np
import random
import threading
import time
import warnings

import keras

from ..utils.anchors import anchor_targets_bbox, bbox_transform
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb


class Generator(object):
    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
    ):
        self.transform_generator  = transform_generator
        self.batch_size           = int(batch_size)
        self.group_method         = group_method
        self.shuffle_groups       = shuffle_groups
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations):
        # randomly transform both image and annotations
        if self.transform_generator:
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def anchor_targets(
        self,
        image_shape,
        annotations,
        num_classes,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs
    ):
        return anchor_targets_bbox(image_shape, annotations, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group     = [[]] * self.batch_size
        regression_group = [[]] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], regression_targets, anchors = self.anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)
            num_positive = max(np.sum([np.sum(labels == 1) for labels in labels_group[index]]), 1)

            # append anchor states and normalization value
            for a, at, lg in zip(anchors, regression_targets, labels_group[index]):
                regression_group[index].append(bbox_transform(a, at))

                # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
                # append normalization value (1 / num_positive_anchors)
                regression_group[index][-1] = np.concatenate([
                    regression_group[index][-1],
                    np.max(lg, axis=1, keepdims=True),  # anchor states
                    np.ones((regression_group[index][-1].shape[0], 1)) * num_positive,  # normalization value
                ], axis=1)

            # append normalization value
            for i in range(len(labels_group[index])):
                labels_group[index][i] = np.concatenate([
                    labels_group[index][i],
                    np.ones((labels_group[index][i].shape[0], 1)) * num_positive,  # normalization value
                ], axis=1)

        # use labels and regression to construct batches for P3...P7
        labels_batches     = [np.zeros((self.batch_size,) + lg.shape, dtype=keras.backend.floatx()) for lg in labels_group[0]]
        regression_batches = [np.zeros((self.batch_size,) + r.shape, dtype=keras.backend.floatx()) for r in regression_group[0]]

        # loop over images
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            # loop over P3...P7 for one image
            for i in range(len(labels)):
                # copy data to corresponding batch
                labels_batches[i][index, ...]     = labels[i]
                regression_batches[i][index, ...] = regression[i]

        return regression_batches + labels_batches

    def compute_input_output(self, group):
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)
