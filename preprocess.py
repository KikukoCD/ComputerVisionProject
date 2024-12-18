# %%

# Load stanford
import os
from sklearn.model_selection import train_test_split
import cv2

from config import config


if config.process_stanford:
    print('----- Stanford')

    with open('datasets/Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        print(f'Train files ({len(train_files)})')
        print(f'Train labels ({len(train_labels)})')

    with open('datasets/Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        print(f'Test files ({len(test_files)})')
        print(f'Test labels ({len(test_labels)})')

    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    S40 = 'datasets/Stanford40/data'

    # Add a validation split
    train_files, validation_files, train_labels, validation_labels = train_test_split(
        train_files, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    splits = {
        'test': (test_files, test_labels),
        'train': (train_files, train_labels),
        'validation': (validation_files, validation_labels),
    }

    for split, (inputs, labels) in splits.items():
        items = []
        # Make directory
        os.makedirs(os.path.join(S40, split), exist_ok=True)
        for input, label in zip(inputs, labels):
            image = cv2.imread(f'datasets/Stanford40/JPEGImages/{input}')
            os.makedirs(os.path.join(S40, split, label), exist_ok=True)
            cv2.imwrite(os.path.join(S40, split, label, input), image)

# %%

if config.process_tvhi:
    print('------ TV-HI')

    set_1_indices = [[2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                     [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48],
                     [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50],
                     [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]]
    set_2_indices = [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39],
                     [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50],
                     [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48],
                     [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    print(f'Set 1 to be used for test ({len(set_1)})')
    print(f'Set 1 labels ({len(set_1_label)})')

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    print(f'Set 2 to be used for train and validation ({len(set_2)})')
    print(f'Set 2 labels ({len(set_2_label)})')

    TVHI = 'datasets/TV-HI/tv_human_interactions_videos'
    TVHI_OUTPUT = 'datasets/TV-HI/data'
    TVHI_OUTPUT_DIRS = [f'{TVHI_OUTPUT}_{i}' for i in range(config.bake_flow_frames)]
    TVHI_OUTPUT_DIRS.append(TVHI_OUTPUT)

    # Add a validation split
    train_files, validation_files, train_labels, validation_labels = train_test_split(
        set_2, set_2_label, test_size=0.1, random_state=42, stratify=set_2_label
    )

    splits = {
        'test': (set_1, set_1_label),
        'train': (train_files, train_labels),
        'validation': (validation_files, validation_labels),
    }

    for split, (inputs, labels) in splits.items():
        print(f'Processing split: {split}')
        items = []
        # Make directory
        os.makedirs(os.path.join(TVHI_OUTPUT_DIRS[-1], split), exist_ok=True)
        # Save the middle frame
        for input, label in zip(inputs, labels):
            cap = cv2.VideoCapture(os.path.join(TVHI, input))

            # Read all frames
            frames = []
            success = True
            while success:
                success, frame = cap.read()
                if success: frames.append(frame)

            for delta in [0.5, 0.6, 0.7, 0.8]:
                filename = f'{input}_{delta}.png'
                prev_image = None

                frame = int(delta * len(frames))
                if frame + config.bake_flow_frames + 1 > len(frames):
                    continue

                for i in range(config.bake_flow_frames + 1):
                    image = frames[frame + i]

                    # Optical flow frames may be without color
                    if i < config.bake_flow_frames: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    os.makedirs(os.path.join(TVHI_OUTPUT_DIRS[i], split, label), exist_ok=True)
                    cv2.imwrite(os.path.join(TVHI_OUTPUT_DIRS[i], split, label, filename), image)

    # %%
