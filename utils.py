from typing import NoReturn, Dict, Optional, List
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow_datasets.core import decode
from tensorflow_datasets.core.utils import type_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from pandas_ml import ConfusionMatrix
from tqdm import tqdm
import wandb


def ds_preview(ds: tf.data.Dataset):
    # Visualize a few images
    sample_images, *other = next(iter(ds))
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(sample_images[:9]):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("int"))
        plt.axis("off")
    plt.show()


def ds_of_preview(ds: tf.data.Dataset, k=3):
    # Visualize a few images
    ofs_batch, *other = next(iter(ds))
    images = None
    if isinstance(ofs_batch, dict):
        ofs_batch, images = ofs_batch['pofinput'], ofs_batch['piminput']
    plt.figure(figsize=(10, 10))
    for i, ofs in enumerate(ofs_batch[:k]):
        if images is not None:
            im = images[i]
            fig, ax = plt.subplots(figsize=plt.figaspect(im))
            fig.subplots_adjust(0, 0, 1, 1)
            plt.imshow(im)
            plt.axis("off")
            plt.show()

        ofs = np.split(ofs, ofs.shape[2] / 2, axis=2)
        image = np.vstack([np.concatenate([of, np.zeros((of.shape[0], of.shape[1], 1))], axis=2) for of in ofs])
        fig, ax = plt.subplots(figsize=plt.figaspect(image))
        fig.subplots_adjust(0, 0, 1, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    plt.show()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# I hope this doesn't get depricated in like 2 days :S
class OpticalFlowDatasetBuilder(tfds.ImageFolder):
    def __init__(
            self, root_dir: str, of_dir: str, *,
            shape: Optional[type_utils.Shape] = None, dtype: Optional[tf.DType] = None
    ):
        self.of_dir = of_dir
        super().__init__(root_dir, shape=shape, dtype=dtype)

    def _download_and_prepare(self, **kwargs) -> NoReturn:
        return self._download_and_prepare()

    def _as_dataset(self, split: str, shuffle_files: bool = False, decoders: Optional[Dict[str, decode.Decoder]] = None,
                    read_config=None) -> tf.data.Dataset:
        ds = super()._as_dataset(split, shuffle_files, decoders, read_config)

        data_dir = self.data_dir
        of_dir = self.of_dir

        # Loads optical flow numpy files
        def _load_of_example(image_path):
            of_path = os.path.join(
                of_dir, os.path.relpath(image_path.numpy().decode(), data_dir)).rsplit('.', 1)[0] + '.npz'
            of = np.load(of_path)['optical_flow']
            return tf.convert_to_tensor(of, dtype=tf.float16)

        return ds \
            .map(lambda row: (row['image'], row['label'], row['image/filename']), num_parallel_calls=tf.data.AUTOTUNE) \
            .map(
            lambda image, label, image_path: (image, tf.py_function(_load_of_example, [image_path], tf.float16), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )


def evaluate_model(model, test_ds: tf.data.Dataset, labels: List[str], config):
    print('Evaluating the model')
    ys = []
    preds = []
    for X, y in tqdm(test_ds):
        pred = model.predict(X)
        ys.extend(y.numpy())
        preds.extend(np.argmax(pred, axis=1))

    print('Generating confusion matrix')
    cm = ConfusionMatrix(ys, preds, labels=labels)
    stats = cm.stats()

    # Format class stats
    class_stats = stats['class']
    class_stats['average'] = stats['class'].mean(axis=1)
    class_stats.to_csv('stats_class.csv')

    # Overall stats
    overall_stats = pd.DataFrame(stats['overall']).transpose()
    overall_stats.to_csv('stats_overall.csv')

    # Create confusion matrix plot
    cm.print_stats()
    cm.plot(max_colors=1, cmap='Greens')
    plt.savefig('stats_cf.png', bbox_inches='tight', pad_inches=1)
    plt.show()

    # Upload results to wandb
    if config.use_wandb:
        wandb.save('stats_class.csv')
        wandb.save('stats_overall.csv')
        wandb.save('stats_cf.png')
        wandb.log({'test_accuracy': stats['overall']['Accuracy']})


# Credit to Brad Kenstler, source: https://github.com/bckenstler/CLR
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', log_wandb=False):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.log_wandb = log_wandb
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        self.i = 0

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())
        if self.log_wandb:
            wandb.log({'lr': self.model.optimizer.lr})

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
        if self.log_wandb and (int(self.clr_iterations) % 20) == 0:
            wandb.log({'lr': self.model.optimizer.lr})
