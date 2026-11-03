"""Shared CSV and TXT serialization for completed evaluation results."""

import csv
from pathlib import Path
from typing import Any


CLASS_NAMES = ("gebrochen", "intakt")


def format_export_metric(value: object) -> str:
    """Format a metric consistently, including undefined values."""
    if value is None:
        return "N/A"
    numeric_value = float(value)
    return f"{numeric_value:.6f} ({numeric_value * 100:.2f}%)"


def relative_test_path(image_path: Path, test_folder: Path) -> str:
    """Return a test-folder-relative path, falling back to the filename."""
    try:
        return image_path.relative_to(test_folder).as_posix()
    except ValueError:
        return image_path.name


def write_evaluation_csv(destination: Path, result: dict[str, Any]) -> None:
    """Write a completed evaluation in the established CSV report format."""
    metrics = result["metrics"]
    matrix = metrics["confusion_matrix"]
    evaluated_at = result["evaluated_at"].isoformat(timespec="seconds")
    test_folder = Path(result["test_folder"])
    metric_rows = (
        ("Richtig-negativ (r_n)", metrics["tn"]),
        ("Falsch-negativ (f_n)", metrics["fn"]),
        ("Falsch-positiv (f_p)", metrics["fp"]),
        ("Richtig-positiv (r_p)", metrics["tp"]),
        ("Genauigkeit", format_export_metric(metrics["accuracy"])),
        ("Sensitivitaet", format_export_metric(metrics["sensitivity"])),
        ("Spezifitaet", format_export_metric(metrics["specificity"])),
        ("Falsch-Positiv-Rate", format_export_metric(metrics["false_positive_rate"])),
        ("Falsch-Negativ-Rate", format_export_metric(metrics["false_negative_rate"])),
    )
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Metadaten der Auswertung"])
        writer.writerow(["Ausgewaehltes Modell", result["model_path"]])
        writer.writerow(["Unabhaengiger Testordner", result["test_folder"]])
        writer.writerow(["Datum und Uhrzeit", evaluated_at])
        writer.writerow(["Klassenreihenfolge", "gebrochen=0", "intakt=1"])
        writer.writerow(["Positive Klasse", "gebrochen (p)"])
        writer.writerow([])
        writer.writerow(["Vorhersagen pro Testbild"])
        writer.writerow([
            "Filename or relative path",
            "True label",
            "Predicted label",
            "Probability gebrochen",
            "Probability intakt",
            "Correctness",
        ])
        for record in result["records"]:
            probabilities = record["probabilities"]
            is_correct = int(record["true_label"]) == int(record["predicted_label"])
            writer.writerow([
                relative_test_path(Path(record["path"]), test_folder),
                CLASS_NAMES[int(record["true_label"])],
                CLASS_NAMES[int(record["predicted_label"])],
                f"{float(probabilities[0]):.8f}",
                f"{float(probabilities[1]):.8f}",
                "Correct" if is_correct else "Incorrect",
            ])
        writer.writerow([])
        writer.writerow(["Konfusionsmatrix (p = gebrochen, n = intakt)"])
        writer.writerow(["Ergebnis", "Anzahl"])
        writer.writerow(["Richtig-negativ (r_n)", matrix[1][1]])
        writer.writerow(["Falsch-negativ (f_n)", matrix[0][1]])
        writer.writerow(["Falsch-positiv (f_p)", matrix[1][0]])
        writer.writerow(["Richtig-positiv (r_p)", matrix[0][0]])
        writer.writerow([])
        writer.writerow(["Kennzahlen"])
        writer.writerows(metric_rows)


def write_evaluation_text(destination: Path, result: dict[str, Any]) -> None:
    """Write a completed evaluation in the established TXT report format."""
    metrics = result["metrics"]
    matrix = metrics["confusion_matrix"]
    test_folder = Path(result["test_folder"])
    lines = [
        "Auswertung des unabhaengigen Tests",
        "===================================",
        f"Ausgewaehltes Modell: {result['model_path']}",
        f"Unabhaengiger Testordner: {result['test_folder']}",
        f"Datum und Uhrzeit: {result['evaluated_at'].isoformat(timespec='seconds')}",
        "Klassenreihenfolge: gebrochen=0, intakt=1",
        "Positive Klasse: gebrochen (p)",
        "",
        "Vorhersagen pro Testbild",
        "-------------------------",
        "Datei | Wahres Label | Vorhersage | P(gebrochen) | P(intakt) | Korrektheit",
    ]
    for record in result["records"]:
        probabilities = record["probabilities"]
        is_correct = int(record["true_label"]) == int(record["predicted_label"])
        lines.append(
            f"{relative_test_path(Path(record['path']), test_folder)} | "
            f"{CLASS_NAMES[int(record['true_label'])]} | "
            f"{CLASS_NAMES[int(record['predicted_label'])]} | "
            f"{float(probabilities[0]):.8f} | {float(probabilities[1]):.8f} | "
            f"{'Korrekt' if is_correct else 'Abweichung'}"
        )
    lines.extend([
        "",
        "Konfusionsmatrix (p = gebrochen, n = intakt)",
        f"Richtig-negativ (r_n): {matrix[1][1]}",
        f"Falsch-negativ (f_n): {matrix[0][1]}",
        f"Falsch-positiv (f_p): {matrix[1][0]}",
        f"Richtig-positiv (r_p): {matrix[0][0]}",
        "",
        f"Genauigkeit: {format_export_metric(metrics['accuracy'])}",
        f"Sensitivitaet: {format_export_metric(metrics['sensitivity'])}",
        f"Spezifitaet: {format_export_metric(metrics['specificity'])}",
        f"Falsch-Positiv-Rate: {format_export_metric(metrics['false_positive_rate'])}",
        f"Falsch-Negativ-Rate: {format_export_metric(metrics['false_negative_rate'])}",
        "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
