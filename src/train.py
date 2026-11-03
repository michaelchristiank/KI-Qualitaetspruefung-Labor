"""Train a simple CNN to distinguish broken and intact chickpeas.

This script deliberately uses only the training and validation data.  The test
data is kept separate so it can later be used for an independent evaluation.
"""

from pathlib import Path
import math
import os
import random
from typing import Callable

import tensorflow as tf

# Use one fixed seed for reproducible TensorFlow and Keras results.
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# 1. Finding the project folders
# The paths are calculated relative to this file, so the script can be started
# from the project root or from another working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
VALIDATION_DIR = PROJECT_DIR / "data" / "validation"
MODEL_PATH = PROJECT_DIR / "models" / "quality_model.keras"

# All images are resized to the same dimensions before entering the network.
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
BATCH_SIZE = 32
# EarlyStopping can end the experiment before all 30 epochs are used.
EPOCHS = 30
CLASS_NAMES = ["gebrochen", "intakt"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# 2. Loading the image from datasets
def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load labelled images from the train and validation folder trees."""
    # labels="inferred" gets each label from its subfolder name.  Supplying
    # class_names fixes the class order: gebrochen=0 and intakt=1.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        pad_to_aspect_ratio=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        pad_to_aspect_ratio=True,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Prefetching prepares the next batch while the model processes the current
    # one. It improves throughput without changing the data or the results.
    autotune = tf.data.AUTOTUNE
    return (
        train_dataset.prefetch(autotune),
        validation_dataset.prefetch(autotune),
    )

# 3. Creating the Convolutional Neural Network (CNN) model for image classification
def create_model(learning_rate: float = 0.001) -> tf.keras.Model:
    """Create a small convolutional neural network for the two classes."""
    model = tf.keras.Sequential(
        [
            # Pixel values normally range from 0 to 255. Rescaling maps them
            # to 0 to 1, which makes neural-network training more stable.
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            # 4.Two output values: one for "gebrochen" and one for "intakt".
            tf.keras.layers.Dense(2, activation="softmax"),
        ],
        name="quality_inspection_cnn",
    )
    # Defines how the model learns
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _validate_gui_training_inputs(
    broken_dir: Path,
    intact_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_output_path: Path,
) -> tuple[list[Path], list[Path], Path]:
    """Validate GUI inputs and return sorted files and a safe model path."""
    class_folders = (("gebrochen", broken_dir), ("intakt", intact_dir))
    images_by_class: list[list[Path]] = []
    for class_name, folder in class_folders:
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Der Ordner für {class_name} ist nicht vorhanden oder ungültig: {folder}")
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 2:
            raise ValueError(
                f"Der Ordner für {class_name} benötigt mindestens zwei unterstützte Bilder "
                f"(gefunden: {len(images)})."
            )
        images_by_class.append(images)

    shared_images = {path.resolve() for path in images_by_class[0]} & {
        path.resolve() for path in images_by_class[1]
    }
    if shared_images:
        raise ValueError("Die beiden Klassenordner enthalten dieselben Bilddateien.")

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Epochen muss eine positive ganze Zahl sein.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Batchgröße muss eine positive ganze Zahl sein.")
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("Die Lernrate muss eine Zahl größer als 0 sein.") from error
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Die Lernrate muss eine endliche Zahl größer als 0 sein.")

    output_path = Path(model_output_path).expanduser()
    if output_path.suffix.lower() != ".keras":
        raise ValueError("Der Modellpfad muss auf .keras enden.")
    if not output_path.parent.is_dir():
        raise ValueError(f"Der Zielordner für das Modell existiert nicht: {output_path.parent}")
    if not os.access(output_path.parent, os.W_OK):
        raise ValueError(f"Im Zielordner kann kein Modell gespeichert werden: {output_path.parent}")
    if output_path.is_dir():
        raise ValueError(f"Der Modellpfad verweist auf einen Ordner: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"Die Modelldatei existiert bereits und wird nicht überschrieben: {output_path}")

    return images_by_class[0], images_by_class[1], output_path


def train_model_for_gui(
    broken_dir: Path,
    intact_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_output_path: Path,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> dict[str, object]:
    """Train from selected class folders using a seeded, stratified 80/20 split.

    The caller may run this function in a worker thread. The callback runs on
    that same worker thread and must therefore communicate with a GUI through
    a thread-safe queue rather than updating widgets directly.
    """
    broken_images, intact_images, output_path = _validate_gui_training_inputs(
        broken_dir, intact_dir, epochs, batch_size, learning_rate, model_output_path
    )
    learning_rate = float(learning_rate)

    rng = random.Random(SEED)
    training_paths: list[str] = []
    training_labels: list[int] = []
    validation_paths: list[str] = []
    validation_labels: list[int] = []

    # Split each class independently so both subsets retain both labels.
    for class_index, class_images in enumerate((broken_images, intact_images)):
        shuffled_images = list(class_images)
        rng.shuffle(shuffled_images)
        validation_count = max(1, int(round(len(shuffled_images) * 0.2)))
        validation_count = min(validation_count, len(shuffled_images) - 1)
        validation_images = shuffled_images[:validation_count]
        training_images = shuffled_images[validation_count:]
        training_paths.extend(str(path) for path in training_images)
        training_labels.extend([class_index] * len(training_images))
        validation_paths.extend(str(path) for path in validation_images)
        validation_labels.extend([class_index] * len(validation_images))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize_with_pad(
            image,
            target_height=IMAGE_HEIGHT,
            target_width=IMAGE_WIDTH,
            method="bilinear",
        )
        return image, label

    autotune = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((training_paths, training_labels))
    train_dataset = train_dataset.shuffle(
        len(training_paths), seed=SEED, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.map(load_image, num_parallel_calls=autotune)
    train_dataset = train_dataset.batch(batch_size).prefetch(autotune)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_paths, validation_labels))
    validation_dataset = validation_dataset.map(load_image, num_parallel_calls=autotune)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(autotune)

    model = create_model(learning_rate=learning_rate)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    **{name: float(value) for name, value in (logs or {}).items()},
                })

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, ProgressCallback()],
    )
    return {
        "model_path": output_path,
        "history": history.history,
        "completed_epochs": len(history.epoch),
        "training_images": len(training_paths),
        "validation_images": len(validation_paths),
    }


def main() -> None:
    """Train the model, validate it after every epoch, and save the best one."""
    train_dataset, validation_dataset = load_datasets()
    model = create_model()

    # The model folder must exist before ModelCheckpoint can write the file.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Stop if validation loss does not improve for five consecutive epochs.
    # restore_best_weights=True returns the model to its best validation state.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # Save only the best complete local Keras model during training.
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    model.summary()
    # fit() prints the loss and accuracy for training and validation per epoch.
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, model_checkpoint],
    )

    # The last values correspond to the final completed epoch.
    final_training_accuracy = history.history["accuracy"][-1]
    final_validation_accuracy = history.history["val_accuracy"][-1]
    best_validation_accuracy = max(history.history["val_accuracy"])
    print(f"Final training accuracy: {final_training_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
