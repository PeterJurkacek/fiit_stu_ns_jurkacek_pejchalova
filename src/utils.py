# %%
from pathlib import Path

import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_loaded_data(labeled_ds, number_of_images=2):
    print(f"Loaded_number_of_images: {number_of_images}")
    for image_raw, label_text in labeled_ds.take(number_of_images):
        print(repr(image_raw.numpy()))
        print()
        print(label_text.numpy())


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_dirs(path: Path):
    dirs = [item.stem for item in path.iterdir() if item.is_dir()]
    print(f"DIRS: {dirs} on PATH: {path}")
    return dirs


def count_dirs(path: Path):
    count = len(get_dirs(path))
    print(f"COUNT: {count}")
    return count


def images_count(path: Path):
    count = len(list(jpg_images_from(path)))
    print(f"IMAGE_COUNTS: {count} on PATH: {path}")
    return count


def jpg_images_from(path: Path):
    return path.glob('**/*.jpg')


def calculate_steps_per_epoch(total_num_of_samples, batch_size):
    steps_per_epoch = total_num_of_samples // batch_size
    print(f'STEPS_PER_EPOCH: {steps_per_epoch}, TOTAL_NUM_OF_SAMPLES: {total_num_of_samples}, BATCH_SIZE: {batch_size}')
    return steps_per_epoch
