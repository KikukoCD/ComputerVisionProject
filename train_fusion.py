import numpy as np
import tensorflow as tf
from model import FramesModel, FusionModel
from config import config, apply_nvidia_optimizations, init_callbacks, save_model
from model_architectures import FRAMES, FUSION_MODEL
from utils import OpticalFlowDatasetBuilder, ds_of_preview, evaluate_model
from tensorflow.keras.optimizers import Adam

# Nvidia Performance guide  advises f16
apply_nvidia_optimizations()

# Initialize callbacks
CALLBACKS = init_callbacks()


# Dataset pipeline
def ds_pipeline(ds: tf.data.Dataset, batch_size):
    return ds \
        .map(lambda image, of, label: (dict(piminput=image, pofinput=of), label), num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(batch_size * 2) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)


builder = OpticalFlowDatasetBuilder(config.tvhi_path, config.tvhi_path + '_of')
labels = builder.info.features['label'].names
print(f'Labels: {labels}')
train_ds, val_ds, test_ds = builder.as_dataset(split=['train', 'validation', 'test'], shuffle_files=True)
train_ds, val_ds, test_ds = ds_pipeline(train_ds, config.batch_size), ds_pipeline(val_ds, config.batch_size), \
                            ds_pipeline(test_ds, config.batch_size)
ds_of_preview(train_ds)

# Build frames model
frames_model = FramesModel(
    input_size=(config.image_size, config.image_size, 3),
    n_classes=False,
    blocks=FRAMES[config.fusion_frames_model_name],
    name='frames_model',
    weights_path=config.fusion_frames_weights,
)

# Build of model
of_model = FramesModel(
    input_size=(config.image_size, config.image_size, config.bake_flow_frames * 2),
    n_classes=False,
    blocks=FRAMES[config.fusion_of_model_name],
    name='of_model',
    weights_path=config.fusion_of_weights,
)

# Build a fused model
model = FusionModel(
    frames_model=frames_model,
    of_model=of_model,
    n_classes=config.tvhi_class_count,
    name='fusion_model',
    blocks=FUSION_MODEL[config.model_name],
    weights_path=config.load_weights
)

# Visualize model
print(model.summary())
tf.keras.utils.plot_model(model, to_file=f'model_{config.model_name}_of.png', show_shapes=True)

# Train model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=config.lr if config.lr_mode != 'cyclic' else config.min_lr),
    metrics=["accuracy"]
)
if config.train:
    model.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        callbacks=CALLBACKS,
        epochs=int(np.ceil(config.epochs / config.epoch_scale)),
        steps_per_epoch=int(np.ceil(len(train_ds) * config.epoch_scale)),
    )
    save_model(model)

if config.evaluate:
    evaluate_model(model, test_ds, labels, config)
