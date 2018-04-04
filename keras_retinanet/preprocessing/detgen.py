from .generator import Generator
from ..utils.anchors import anchor_targets_bbox, bbox_transform


class DetDataGenerator(Generator):
    def __init__(self, detgen, augmenter, **kwargs):
        self.classes = detgen.classes
        self.retina_comp_generator = detgen.get_retina_comp_generator(augmenter=augmenter)
        self.transform_generator = None
        self.image_min_side = 750
        self.image_max_side = 750
        self.batch_size = detgen.batch_size
        self.detgen = detgen
        self.compute_anchor_targets = anchor_targets_bbox

    def num_classes(self):
        return len(self.detgen.classes)

    def size(self):
        return len(self.detgen.index_df)

    def image_aspect_ratio(self, image_index):
        return None

    def compute_input_output(self, group):
        # load images and annotations
        image_group, annotations_group = group

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
        image_group, annotations_group = next(self.retina_comp_generator)
        return self.compute_input_output((image_group, annotations_group))
