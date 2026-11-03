"""Classify one local image with the trained quality-inspection model.

Example after training:
    python src/inference.py path/to/your/image.jpg
"""

from pathlib import Path
import sys

import tensorflow as tf

from preprocessing import load_image_for_model


# These values must remain identical to the values in train.py.
IMAGE_HEIGHT = 180
IMAGE_WIDTH = 180
CLASS_NAMES = ["gebrochen", "intakt"]

# Calculate the model location relative to this source file. This works even
# when the script is started from a different working directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "quality_model.keras"


def load_image(image_path: Path) -> tf.Tensor:
    """Load and pad one image exactly like the training input pipeline."""
    return load_image_for_model(image_path)


def main() -> None:
    """Load the local model, classify one local image, and print the result."""
    if len(sys.argv) != 2:
        print("Usage: python src/inference.py <path-to-image>")
        print("Please provide the path to an existing image file.")
        return

    image_path = Path(sys.argv[1])

    # Check that the trained model is available.
    if not MODEL_PATH.is_file():
        print(f"Error: Model file not found: {MODEL_PATH}")
        print("Please train the model before using inference.")
        return

    if not image_path.is_file():
        print(f"Error: Image file not found: {image_path}")
        print("Please provide the path to an existing image file.")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        image_batch = load_image(image_path)
        probabilities = model.predict(image_batch, verbose=0)[0]
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"Error: Could not load or classify the image: {error}")
        return

    # The output order is guaranteed by CLASS_NAMES, which is also used in
    # train.py: index 0 is gebrochen and index 1 is intakt.
    print("Probabilities:")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"  {class_name}: {probability * 100:.2f}%")

    predicted_index = int(tf.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    print(f"Final classification: {predicted_class}")


if __name__ == "__main__":
    main()
