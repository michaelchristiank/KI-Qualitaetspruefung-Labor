"""Create reproducible train, validation, and test splits from local raw data.
The raw images are never changed. This script copies them into the generated
dataset folders so that training, validation, and testing use separate files."""

from pathlib import Path
import random
import shutil
# Use the same class names as the other project scripts.
CLASS_NAMES = ["gebrochen", "intakt"]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}

# Expected number of images for each class.
EXPECTED_IMAGES_PER_CLASS = {
    "gebrochen": 260,
    "intakt": 280,
}

# Dataset split percentages.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# A fixed seed makes a later run create exactly the same split again,
# as long as the images in data/raw have not changed.
RANDOM_SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"

SPLIT_DIRECTORIES = {
    "train": PROJECT_DIR / "data" / "train",
    "validation": PROJECT_DIR / "data" / "validation",
    "test": PROJECT_DIR / "data" / "test",
}

def find_images(folder: Path) -> list[Path]:
    """Return supported image files in a folder, ordered for reproducibility."""

    if not folder.is_dir():
        return []

    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )

def calculate_split_sizes(total_images: int) -> tuple[int, int, int]:
    """Calculate train, validation, and test sizes for one class."""

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    # Give all remaining images to the test set.
    # This guarantees that the three numbers always add up
    # exactly to total_images.
    test_count = total_images - train_count - validation_count

    return train_count, validation_count, test_count

def clear_generated_images(target_folder: Path) -> None:
    """Delete only supported image files from one generated class folder."""

    target_folder.mkdir(parents=True, exist_ok=True)

    # Do not delete directories or files with unsupported extensions.
    # This protects any notes or other non-image files.
    for path in target_folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path.unlink()

def destination_for(source: Path, target_folder: Path) -> Path:
    """Choose a target name without overwriting another file."""

    candidate = target_folder / source.name

    number = 1

    while candidate.exists():

        candidate = target_folder / f"{source.stem}_{number}{source.suffix}"

        number += 1

    return candidate

def copy_images(images: list[Path], target_folder: Path) -> int:
    """Copy image files to one split folder and return the copied count."""

    copied_count = 0

    for image_path in images:

        # copy2() copies the image and preserves local file metadata when able.
        shutil.copy2(
            image_path,
            destination_for(image_path, target_folder)
        )

        copied_count += 1

    return copied_count

def main() -> None:
    """Validate raw data, create splits, and print a clear local summary."""

    raw_images: dict[str, list[Path]] = {}
    invalid_classes = []
    for class_name in CLASS_NAMES:

        class_images = find_images(
            RAW_DIR / class_name
        )

        raw_images[class_name] = class_images

        expected_count = EXPECTED_IMAGES_PER_CLASS[class_name]

        if len(class_images) != expected_count:

            invalid_classes.append(
                (
                    class_name,
                    len(class_images),
                    expected_count,
                )
            )

    if invalid_classes:

        for class_name, image_count, expected_count in invalid_classes:

            print(
                f"Error: '{RAW_DIR / class_name}' contains "
                f"{image_count} supported JPEG/JPG image(s); "
                f"expected exactly {expected_count}."
            )

        print(
            "Generated train/validation/test images were not changed."
        )

        return

    summary: dict[str, dict[str, int]] = {
        split_name: {}
        for split_name in SPLIT_DIRECTORIES
    }

    prepared_splits: dict[str, dict[str, list[Path]]] = {
        split_name: {}
        for split_name in SPLIT_DIRECTORIES
    }

    for class_index, class_name in enumerate(CLASS_NAMES):

        # Make a copy so the original list is not changed.
        images = raw_images[class_name].copy()

        # Use a different but reproducible seed for each class.
        random.Random(
            RANDOM_SEED + class_index
        ).shuffle(images)

        # Calculate split sizes according to the class size.
        (
            train_count,
            validation_count,
            test_count,
        ) = calculate_split_sizes(len(images))

        prepared_splits["train"][class_name] = (
            images[:train_count]
        )

        prepared_splits["validation"][class_name] = (
            images[
                train_count:
                train_count + validation_count
            ]
        )

        prepared_splits["test"][class_name] = (
            images[
                train_count + validation_count:
                train_count + validation_count + test_count
            ]
        )

    # This happens only after all raw-data checks and split
    # calculations have succeeded.
    for split_name, split_directory in SPLIT_DIRECTORIES.items():

        for class_name in CLASS_NAMES:

            clear_generated_images(
                split_directory / class_name
            )

    for split_name, split_directory in SPLIT_DIRECTORIES.items():

        for class_name in CLASS_NAMES:

            images_for_split = prepared_splits[
                split_name
            ][class_name]

            summary[split_name][class_name] = copy_images(
                images_for_split,
                split_directory / class_name,
            )

    print()
    print("Dataset split completed successfully.")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    print("Split ratios:")
    print(f"  Train:      {TRAIN_RATIO * 100:.0f}%")
    print(f"  Validation: {VALIDATION_RATIO * 100:.0f}%")
    print(f"  Test:       {TEST_RATIO * 100:.0f}%")

    print()
    print("Images per class and split:")

    print(
        f"{'Class':<14}"
        f"{'Raw':>6}"
        f"{'Train':>8}"
        f"{'Validation':>13}"
        f"{'Test':>8}"
    )

    print("-" * 55)

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:<14}"
            f"{len(raw_images[class_name]):>6}"
            f"{summary['train'][class_name]:>8}"
            f"{summary['validation'][class_name]:>13}"
            f"{summary['test'][class_name]:>8}"
        )

    total_raw = sum(
        len(raw_images[class_name])
        for class_name in CLASS_NAMES
    )

    total_train = sum(
        summary["train"][class_name]
        for class_name in CLASS_NAMES
    )

    total_validation = sum(
        summary["validation"][class_name]
        for class_name in CLASS_NAMES
    )

    total_test = sum(
        summary["test"][class_name]
        for class_name in CLASS_NAMES
    )

    print("-" * 55)

    print(
        f"{'TOTAL':<14}"
        f"{total_raw:>6}"
        f"{total_train:>8}"
        f"{total_validation:>13}"
        f"{total_test:>8}"
    )

if __name__ == "__main__":
    main()