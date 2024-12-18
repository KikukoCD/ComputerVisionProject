from pathlib import Path
import cv2
import os
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np

from config import config
from utils import chunks

DEBUG = False

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
AUGMENT_TRANSFORM = [
    # horizontally flip 50% of all images
    iaa.Fliplr(0.5),
    # crop images by -5% to 10% of their height/width
    sometimes(iaa.Crop(
        percent=(0, 0.1),
    )),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45),  # rotate by -45 to +45 degrees
        shear=(-16, 16),  # shear by -16 to +16 degrees
        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
]
AUGMENT_NONLINEAR_TRANSFORM = [
    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    # move pixels locally around (with random strengths)
    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
]
AUGMENT_GREYSCALE = [
    # convert images into their superpixel representation
    iaa.OneOf([
        iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 2.0
        iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
        iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
    ]),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
    # search either for all edges or for directed edges,
    # blend the result with the original image using a blobby mask
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # add gaussian noise to images
    iaa.OneOf([
        iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    ]),
    # iaa.Invert(0.05, per_channel=True),  # invert color channels
    iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    # either change the brightness of the whole image (sometimes
    # per channel) or change the brightness of subareas
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
]

AUGMENT_COLOR = [
    *AUGMENT_GREYSCALE,
    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    iaa.Grayscale(alpha=(0.0, 1.0)),
]

AUGMENT_FRAME_FLOW = iaa.Sequential([
    *AUGMENT_TRANSFORM,
    iaa.SomeOf((0, 5), [
        *AUGMENT_COLOR,
        *AUGMENT_NONLINEAR_TRANSFORM,
    ], random_order=True)
])

AUGMENT_OPTICAL_FLOW = iaa.Sequential([
    *AUGMENT_TRANSFORM,
    iaa.SomeOf((0, 5), [
        *AUGMENT_GREYSCALE,
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
    ], random_order=True)
])

AUGMENT_COLOR_FLOW = iaa.Sequential([
    iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20))),
    iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0)))
])


# Extracts optical flow from an images and scales it by 1/5th as normalization
def optical_flow_extract(prev, curr):
    return cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        pyr_scale=0.75, levels=1, winsize=7, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    ) / 3.


# Visualizes an image and its optical flow frames
def plot_img_optical_flow(img, ofs, of=True):
    fig, ax = plt.subplots(figsize=plt.figaspect(img))
    fig.subplots_adjust(0, 0, 1, 1)
    plt.imshow(img)
    plt.show()
    fig, ax = plt.subplots(figsize=plt.figaspect(np.vstack(ofs)))
    fig.subplots_adjust(0, 0, 1, 1)
    if of:
        plt.imshow(np.vstack([np.concatenate([of, np.zeros((of.shape[0], of.shape[1], 1))], axis=2) for of in ofs]))
    else:
        plt.imshow(np.vstack(ofs))
    plt.show()


def augment_frames(augment=True):
    # When augmenting we use any transformation and effect and resize at the last step
    AUGMENT_FLOW = iaa.Sequential([
        # iaa.Resize((config.image_size, config.image_size)),
        *([AUGMENT_FRAME_FLOW] if augment else []),
        iaa.OneOf([
            iaa.CenterCropToAspectRatio(1),
            iaa.PadToAspectRatio(1),
        ]),
        iaa.Resize((config.image_size, config.image_size)),
    ])
    return lambda images, **kwargs: AUGMENT_FLOW(images=images)


def augment_opticalflow(augment=True):
    # When augmenting we use any "affine" transformation and effect and resize at the last step
    AUGMENT_FLOW = iaa.Sequential([
        *([AUGMENT_OPTICAL_FLOW] if augment else []),
        iaa.OneOf([
            iaa.CenterCropToAspectRatio(1),
            iaa.PadToAspectRatio(1),
        ]),
        iaa.Resize((config.image_size, config.image_size)),
    ])

    def augment_fn(images, lagged_images):
        # Generate a deterministic augmenter to identically augment both image and it's timeline
        AUGMENT_FLOW.seed_()
        aug = AUGMENT_FLOW.to_deterministic()
        images = AUGMENT_COLOR_FLOW(images=aug(images=images)) if augment else aug(images=images)
        lagged_images = [aug(images=group) for group in lagged_images]

        if DEBUG:  # Visualize a few images
            for i, im, ofs in zip(range(6), images, zip(*lagged_images)):
                plot_img_optical_flow(im, ofs, of=False)

        # Calcualte optical flow between consequent pairs of images
        of_pairs = list(zip(lagged_images[:-1], lagged_images[1:]))
        of_images = [list(map(lambda pair: optical_flow_extract(*pair), zip(*of_pair))) for of_pair in of_pairs]
        of_images = list(zip(*of_images))
        if DEBUG:  # Visualize a few images
            for i, im, ofs in zip(range(6), images, of_images):
                plot_img_optical_flow(im, ofs)

        # Combine the optical flows and change type to f16 to save space
        optical_flows = [np.concatenate(group, axis=2).astype(np.float16) for group in of_images]
        return zip(images, optical_flows)

    return augment_fn

flow_dirs = [f'data_{i}' for i in range(config.bake_flow_frames)]
flow_dirs.append('data')

print(f'Baking dataset: {config.bake_input}. Augment ratio: {config.augment_repeat_count}')
p = Path(config.bake_input)
for fold_path in p.glob('*'):
    fold = os.path.relpath(fold_path, p)
    print(f'Processing fold: {fold}')

    # Configure Augmentation
    augment = fold in config.augment_folds
    flow = augment_opticalflow(augment=augment) if config.bake_optical_flow else augment_frames(augment=augment)

    for label_path in fold_path.glob('*'):
        label = os.path.relpath(label_path, fold_path)
        print(f'- Processing label: {label}')

        # Create directory for data
        os.makedirs(os.path.join(config.bake_output, fold, label), exist_ok=True)
        if config.bake_optical_flow:
            os.makedirs(os.path.join(f'{config.bake_output}_of', fold, label), exist_ok=True)

        for batch in chunks(list(label_path.glob('*')), config.batch_size):
            # Load images to augment
            files = list(map(lambda p: os.path.splitext(os.path.basename(p))[0], batch))
            images = list(map(lambda p: cv2.imread(str(p)), batch))
            # Load a timeline of images for optical flow
            lagged_images = [
                list(map(lambda p: cv2.imread(str(p).replace('/data/', f'/{flow_dir}/'), 0), batch))
                for flow_dir in flow_dirs
            ] if config.bake_optical_flow else None

            # Repeat augmentation multiple times to generate different variants
            for i in range(1 if not augment else config.augment_repeat_count):
                augmented = flow(images=images, lagged_images=lagged_images)

                for name, image in zip(files, augmented):
                    # Save optizal flow as a numpy matrix if needed
                    if config.bake_optical_flow:
                        image, of = image
                        cv2.imwrite(os.path.join(config.bake_output, fold, label, f'{name}_{i}.png'), image)
                        np.savez_compressed(
                            os.path.join(f'{config.bake_output}_of', fold, label, f'{name}_{i}.npz'),
                            optical_flow=of
                        )
                    else:
                        cv2.imwrite(os.path.join(config.bake_output, fold, label, f'{name}_{i}.png'), image)
