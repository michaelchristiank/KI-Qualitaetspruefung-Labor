"""Shared local image preprocessing for the quality-inspection model."""

from pathlib import Path

import tensorflow as tf


# These values match the training configuration in train.py.
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
CLASS_NAMES = ["gebrochen", "intakt"]


def load_image_for_model(image_path: Path) -> tf.Tensor:
    """Load one image and pad it proportionally to the model input size.

    The image is resized with bilinear interpolation, while preserving its
    original aspect ratio. Empty areas are padded with zeros, just like
    image_dataset_from_directory(..., pad_to_aspect_ratio=True) in train.py.
    Pixel values remain in the 0-255 range because the saved model already
    contains its own Rescaling(1/255) layer.
    """
    image = tf.keras.utils.load_img(image_path, color_mode="rgb")
    image_array = tf.keras.utils.img_to_array(image)
    padded_image = tf.image.resize_with_pad(
        image_array,
        target_height=IMAGE_HEIGHT,
        target_width=IMAGE_WIDTH,
        method="bilinear",
    )

    # The model expects a batch dimension, even for one selected image.
    return tf.expand_dims(padded_image, axis=0)
