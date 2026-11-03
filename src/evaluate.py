"""Independently evaluate the trained model using only the test dataset.

The test data must never be used while training or choosing model settings.
It is reserved for this final, independent quality check.
"""

from numbers import Integral
from pathlib import Path
from typing import Sequence

import tensorflow as tf


# These values must match train.py and inference.py exactly.
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
CLASS_NAMES = ["gebrochen", "intakt"]

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "data" / "test"
MODEL_PATH = PROJECT_DIR / "models" / "quality_model.keras"
RESULTS_PATH = PROJECT_DIR / "results" / "evaluation_results.txt"
BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpeg", ".jpg"}


def find_test_images() -> list[Path]:
    """Return supported image files below data/test, without loading them."""
    if not TEST_DIR.is_dir():
        return []
    return [
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def format_rate(numerator: int, denominator: int) -> str:
    """Format a rate safely when a class is absent from the test dataset."""
    if denominator == 0:
        return "not defined (no relevant test images)"
    return f"{numerator / denominator:.4f} ({numerator / denominator * 100:.2f}%)"


def calculate_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, object]:
    """Calculate the confusion matrix and binary metrics for class labels.

    Class 0 (gebrochen) is the positive class and class 1 (intakt) is the
    negative class. The confusion matrix is indexed as
    ``[true_class][predicted_class]``. Undefined metrics are returned as None.
    """
    true_values = list(true_labels)
    predicted_values = list(predicted_labels)
    if len(true_values) != len(predicted_values):
        raise ValueError(
            "True and predicted labels must have the same number of values "
            f"(got {len(true_values)} and {len(predicted_values)})."
        )

    for label_set_name, labels in (
        ("true_labels", true_values),
        ("predicted_labels", predicted_values),
    ):
        for index, label in enumerate(labels):
            if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
                raise ValueError(
                    f"{label_set_name}[{index}] must be 0 (gebrochen) or 1 (intakt); "
                    f"got {label!r}."
                )

    true_positive = sum(
        true == 0 and predicted == 0
        for true, predicted in zip(true_values, predicted_values)
    )
    false_negative = sum(
        true == 0 and predicted == 1
        for true, predicted in zip(true_values, predicted_values)
    )
    false_positive = sum(
        true == 1 and predicted == 0
        for true, predicted in zip(true_values, predicted_values)
    )
    true_negative = sum(
        true == 1 and predicted == 1
        for true, predicted in zip(true_values, predicted_values)
    )

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    total = len(true_values)

    return {
        "confusion_matrix": [
            [true_positive, false_negative],
            [false_positive, true_negative],
        ],
        "tp": true_positive,
        "fn": false_negative,
        "fp": false_positive,
        "tn": true_negative,
        "accuracy": (true_positive + true_negative) / total if total else None,
        "sensitivity": true_positive / positive_count if positive_count else None,
        "specificity": true_negative / negative_count if negative_count else None,
        "false_positive_rate": false_positive / negative_count if negative_count else None,
        "false_negative_rate": false_negative / positive_count if positive_count else None,
    }


def main() -> None:
    """Load the local model, predict all test images, and print metrics."""
    if not MODEL_PATH.is_file():
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Train the model with train.py before running this evaluation.")
        return

    # This check gives a simple message before Keras attempts to read folders.
    if not find_test_images():
        print(f"Error: No supported images found in: {TEST_DIR}")
        print("Add test images to data/test/gebrochen and data/test/intakt.")
        return

    try:
        # Labels are inferred from the subfolder names. Setting class_names
        # makes the mapping explicit: gebrochen=0, intakt=1.
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            TEST_DIR,
            labels="inferred",
            label_mode="int",
            class_names=CLASS_NAMES,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            # Match the aspect-ratio padding used for training.
            pad_to_aspect_ratio=True,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        model = tf.keras.models.load_model(MODEL_PATH)
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"Error: Could not load the model or test dataset: {error}")
        return

    # train.py saves a model whose first layer rescales pixels from 0-255 to
    # 0-1. Therefore no additional normalization is needed in this script.
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    for image_batch, label_batch in test_dataset:
        probabilities = model(image_batch, training=False)
        predictions = tf.argmax(probabilities, axis=1)
        true_labels.extend(label_batch.numpy().tolist())
        predicted_labels.extend(predictions.numpy().tolist())

    # Positive class: gebrochen (index 0). Negative class: intakt (index 1).
    # TP: a broken part correctly predicted as broken.
    # TN: an intact part correctly predicted as intact.
    # FP: an intact part incorrectly predicted as broken (false alarm).
    # FN: a broken part incorrectly predicted as intact (missed defect).
    true_positive = sum(
        true == 0 and predicted == 0
        for true, predicted in zip(true_labels, predicted_labels)
    )
    true_negative = sum(
        true == 1 and predicted == 1
        for true, predicted in zip(true_labels, predicted_labels)
    )
    false_positive = sum(
        true == 1 and predicted == 0
        for true, predicted in zip(true_labels, predicted_labels)
    )
    false_negative = sum(
        true == 0 and predicted == 1
        for true, predicted in zip(true_labels, predicted_labels)
    )

    total = len(true_labels)
    accuracy = (true_positive + true_negative) / total
    sensitivity = format_rate(true_positive, true_positive + false_negative)
    specificity = format_rate(true_negative, true_negative + false_positive)
    false_negative_rate = format_rate(false_negative, true_positive + false_negative)
    false_positive_rate = format_rate(false_positive, true_negative + false_positive)

    # Rows are true classes; columns are predicted classes.
    results = [
        "Auswertung des unabhaengigen Tests",
        "Positive Klasse: gebrochen (p) | Negative Klasse: intakt (n)",
        "",
        "Konfusionsmatrix (p = gebrochen, n = intakt):",
        f"Richtig-negativ (r_n): {true_negative}",
        f"Falsch-negativ (f_n): {false_negative}",
        f"Falsch-positiv (f_p): {false_positive}",
        f"Richtig-positiv (r_p): {true_positive}",
        "",
        f"Testbilder: {total}",
        f"Genauigkeit: {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Sensitivitaet: {sensitivity}",
        f"Spezifitaet: {specificity}",
        f"Falsch-Negativ-Rate: {false_negative_rate}",
        f"Falsch-Positiv-Rate: {false_positive_rate}",
    ]
    report = "\n".join(results)
    print(report)

    # Saving a text file is optional for analysis, but needs no extra software.
    try:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(report + "\n", encoding="utf-8")
        print(f"\nResults saved to: {RESULTS_PATH}")
    except OSError as error:
        print(f"\nWarning: Could not save results: {error}")


if __name__ == "__main__":
    main()
