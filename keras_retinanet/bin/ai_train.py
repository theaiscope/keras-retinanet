import keras
import os
from deepsense import neptune
from keras_retinanet.bin.train import create_models
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.resnet import download_imagenet, \
    resnet_retinanet as retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.transform import random_transform_generator

ctx = neptune.Context()
img_min_side = 500
img_max_side = 600

DATA_DIR = '/input/deepfashion_data/'

LABELS = os.path.join(DATA_DIR, 'retina_labels.csv')
TRAIN_DATA = os.path.join(DATA_DIR, 'retina_train_less_negs.csv')
VAL_DATA = os.path.join(DATA_DIR, 'retina_valid.csv')

transform_generator = random_transform_generator(
    min_rotation=-0.1,
    max_rotation=0.1,
    min_translation=(-0.2, -0.2),
    max_translation=(0.2, 0.2),
    min_shear=-0.2,
    max_shear=0.2,
    min_scaling=(0.75, 0.75),
    max_scaling=(-1.5, 1.5),
    flip_x_chance=0.3,
    flip_y_chance=0.3,
)

train_generator = CSVGenerator(
    TRAIN_DATA,
    LABELS,
    base_dir='/input/deepfashion_data',
    batch_size=1,
    image_max_side=img_max_side,
    image_min_side=img_min_side,
    #    transform_generator=transform_generator
)

validation_generator = CSVGenerator(
    VAL_DATA,
    LABELS,
    base_dir='/input/deepfashion_data',
    batch_size=1,
    image_min_side=img_min_side,
    image_max_side=img_max_side
)

weights = download_imagenet('resnet50')

model_checkpoint = keras.callbacks.ModelCheckpoint('/output/mod-{epoch:02d}_loss-{loss:.4f}.h5',
                                                   monitor='loss',
                                                   verbose=2,
                                                   save_best_only=False,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   period=1)

callbacks = []


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        ctx.channel_send("loss", logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        ctx.channel_send("val_loss", logs.get('val_loss'))


evaluation = Evaluate(validation_generator, tensorboard=None)
callbacks.append(keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0
))

callbacks.append(LossHistory())
callbacks.append(model_checkpoint)
callbacks.append(evaluation)

model, training_model, prediction_model = create_models(
    backbone_retinanet=retinanet,
    backbone='resnet50',
    num_classes=train_generator.num_classes(),
    weights=weights,
    multi_gpu=0,
    freeze_backbone=True
)

training_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=5000,
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=100
)