__all__ = ('config', 'apply_nvidia_optimizations', 'init_callbacks', 'save_model')

import os
from dataclasses import dataclass
import datetime as dt

from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable
import tensorflow as tf
import imgaug as ia
import wandb
from utils import CyclicLR


@dataclass
class Config(Serializable):
    # General
    seed: int = 42
    use_wandb: bool = os.getenv('USE_WANDB', False)
    use_tensorboard: bool = os.getenv('PROFILE_TF', False)
    log_frequency: int = 20
    n_cpus = 4
    dump_config: bool = False
    train: bool = True
    evaluate: bool = True

    # Generic Model Params
    run_name: str = 'frames_stanford'
    model_name: str = 'frames_stanford_v2'
    batch_size: int = 128
    image_size: int = 224
    dropout: float = 0.5
    epochs: int = 30
    epoch_scale: float = 0.5
    load_weights: str = False
    freeze_blocks: int = 0
    lr: float = 0.001
    min_lr: float = 1e-7
    max_lr: float = 1e-2
    lr_step_size: int = 700  # Time to complete one lr cycle
    lr_mode: str = 'cyclic'

    # Frames model
    frames_dataset: str = 'stanford'

    # Fusion model
    fusion_frames_model_name: str = 'frames_stanford_v2'
    fusion_frames_weights: str = False # 'snapshots/model_frames_frames_stanford_v2.h5'
    fusion_of_model_name: str = 'frames_stanford_v2'
    fusion_of_weights: str = False # 'snapshots/model_of_frames_stanford_v2.h5'

    # Augmentation Config
    augment_folds = ['train']
    augment_repeat_count: int = 25

    # Processing
    process_stanford: bool = False
    process_tvhi: bool = False

    # Baking Dataset
    bake_input: str = 'datasets/Stanford40/data'
    bake_output: str = 'datasets/Stanford40/baked'
    bake_optical_flow: bool = False
    bake_flow_frames: int = 5

    # Stanford Dataset
    stanford_class_count: int = 40
    stanford_path: str = 'datasets/Stanford40/baked'

    # TVHI Dataset
    tvhi_class_count: int = 4
    tvhi_path: str = 'datasets/TV-HI/baked'


# Try loading overridden config
config: Config = Config.load('config.yml') if os.path.exists('config.yml') else None

# Parse the cli overrides
parser = ArgumentParser()
parser.add_arguments(Config, dest="config", default=config)
args = parser.parse_args()
config: Config = args.config

# Dump the config to a file
if config.dump_config:
    config.save('config.yml')

# Seed everythong
tf.random.set_seed(42)
ia.seed(42)


# Nvidia Performance guide advises f16
def apply_nvidia_optimizations():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)


# Initialize wandb
def init_wandb():
    if config.use_wandb:
        wandb.init(
            project='INFOCV',
            config=config.__dict__
        )


def init_callbacks():
    CALLBACKS = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ]

    # Initialize wandb
    if config.use_wandb:
        init_wandb()
        from wandb.keras import WandbCallback
        CALLBACKS.append(WandbCallback(
            log_batch_frequency=config.log_frequency
        ))

    # Initialize tensorboard for profiling the training flow
    if config.use_tensorboard:
        logs = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        CALLBACKS.append(tf.keras.callbacks.TensorBoard(
            log_dir=logs, histogram_freq=1, profile_batch='2,5'
        ))

    if config.lr_mode == 'cyclic':
        clr = CyclicLR(
            mode='triangular2',
            base_lr=config.min_lr,
            max_lr=config.max_lr,
            step_size=config.lr_step_size,
            log_wandb=config.use_wandb
        )
        CALLBACKS.append(clr)

    return CALLBACKS


def save_model(model):
    print('Saving the best model')
    model.save('best_model_manual.h5')

    if config.use_wandb:
        wandb.save('best_model_manual.h5')
