
# KI-basierte Qualitätssicherung

Offlinefähige Desktop-Anwendung zur KI-gestützten visuellen Qualitätsprüfung von Kichererbsen. Das Projekt wurde im Rahmen einer Studienarbeit im Studiengang Maschinenbau und Produktionstechnik/-management an der HAW Hamburg entwickelt.

Die Anwendung verwendet ein Convolutional Neural Network (CNN), um Bilder in zwei Klassen einzuordnen:

- `gebrochen` – gebrochene Kichererbse
- `intakt` – intakte Kichererbse

Die grafische Benutzeroberfläche (GUI) befindet sich in `src/ui_prototype.py`. Sie ermöglicht das Training eines Modells mit selbst ausgewählten Bildern sowie die anschließende Evaluation anhand eines unabhängigen Testdatensatzes.

## Funktionen

### 1. Modelltraining

Im Tab **Training** können Studierende:

- separate Bildordner für `gebrochen` und `intakt` auswählen;
- die Anzahl der Epochen, Batchgröße und Lernrate einstellen;
- einen Speicherort für das trainierte Modell im `.keras`-Format auswählen;
- den Trainingsfortschritt sowie Trainings- und Validierungskennzahlen verfolgen.

Die Anwendung teilt die Bilder jeder Klasse reproduzierbar in Trainings- und Validierungsdaten auf:

- 80 % Trainingsdaten
- 20 % Validierungsdaten

Für die Aufteilung wird der feste Zufalls-Seed `42` verwendet. Der unabhängige Testdatensatz wird nicht für das Training verwendet.

Das CNN verarbeitet RGB-Bilder mit einer Eingabegröße von 180 × 180 Pixeln. Die Bildvorverarbeitung verwendet Padding und bilineare Interpolation, damit das ursprüngliche Seitenverhältnis erhalten bleibt.

Early Stopping überwacht den Validierungsverlust. Wenn sich dieser fünf Epochen lang nicht verbessert, kann das Training vorzeitig beendet werden. Das Modell mit dem besten Validierungsverlust wird gespeichert.

### 2. Modelltest und Evaluation

Im Tab **Test & Ergebnisse** können Studierende ein trainiertes `.keras`-Modell und einen unabhängigen Testordner auswählen.

Der Testordner muss folgende Struktur haben:

```text
test/
├── gebrochen/
│   ├── bild_001.jpg
│   └── ...
└── intakt/
    ├── bild_001.jpg
    └── ...
```

Die Anwendung zeigt Einzelvorhersagen, eine Konfusionsmatrix und Bewertungskennzahlen an.

Die Klasse `gebrochen` ist als positive Klasse definiert.

Berechnete Kennzahlen:

- **Accuracy (Genauigkeit):** Anteil aller korrekt klassifizierten Testbilder.
- **Sensitivity (Sensitivität):** Anteil der tatsächlich gebrochenen Kichererbsen, die als `gebrochen` erkannt werden.
- **Specificity (Spezifität):** Anteil der tatsächlich intakten Kichererbsen, die als `intakt` erkannt werden.
- **False Positive Rate:** Anteil der intakten Kichererbsen, die fälschlich als `gebrochen` eingestuft werden.
- **False Negative Rate:** Anteil der gebrochenen Kichererbsen, die fälschlich als `intakt` eingestuft werden.

Testbilder dürfen nicht gleichzeitig zum Trainieren oder zur Validierung verwendet werden. Ein unabhängiger Testsatz ist notwendig, um die Modellleistung möglichst unverzerrt zu beurteilen.

### 3. Export

Nach einer erfolgreichen Evaluation können die Auswertungsergebnisse als CSV- oder TXT-Bericht gespeichert werden.

Der Export der Evaluationsberichte ist in der aktuellen GUI an die vorgesehene Sitzungs-/Referenzfreigabe gebunden.

Das trainierte Modell wird bereits beim Training am ausgewählten Speicherort gespeichert. Der separate Button **Trainiertes Modell** im Tab **Export** ist in der aktuellen Version noch nicht implementiert.

## Projektstruktur

```text
KI_Qualitaetspruefung_Labor/
├── assets/
│   ├── admin.png
│   ├── qualitaetspruefung.ico
│   └── qualitaetspruefung.png
├── data/
│   └── test/
│       ├── gebrochen/
│       └── intakt/
|    └── train/
|       ├── gebrochen/
|       └── intakt/
├── models/
│   └── quality_model.keras
├── results/
├── src/
│   ├── evaluate.py
│   ├── evaluation_export.py
│   ├── inference.py
│   ├── prepare_dataset.py
│   ├── preprocessing.py
│   ├── train.py
│   └── ui_prototype.py
├── README.md
└── requirements.txt
```

### Hinweise zur Ordnerstruktur

- `assets/` enthält die verwendeten Symbole und Grafiken der GUI.
- `data/test/` enthält den mitgelieferten unabhängigen Testdatensatz.
- `models/quality_model.keras` enthält ein bereits vorhandenes Beispielmodell.
- `results/` ist für gespeicherte Auswertungsergebnisse vorgesehen.
- `src/` enthält die GUI und die zugehörigen Python-Module.

Die Ordner `data/train/` soll in finale Laborversion nicht enthalten. Für das Training über die GUI wählen die Studierenden ihre Bildordner selbst aus. Die Trainingsfunktion erstellt daraus die Trainings- und Validierungsaufteilung.

`prepare_dataset.py` ist ein separates Hilfsskript, das Rohdaten unter `data/raw/` erwartet. Es wird für den regulären Trainingsablauf über die GUI nicht benötigt.

Das mitgelieferte Modell dient als Beispiel. Beim Training können Studierende einen anderen Speicherort und Dateinamen auswählen.

## Voraussetzungen

- Windows 10 oder Windows 11 (64 Bit)
- Python 3.12
- Abhängigkeiten aus `requirements.txt`

Die Anwendung verwendet:

- TensorFlow für das CNN, Training und die Vorhersagen
- CustomTkinter für die grafische Benutzeroberfläche
- Pillow für die Bilddarstellung

Tkinter gehört zur üblichen Windows-Python-Installation und wird nicht separat über `pip` installiert.

## Installation unter Windows

Öffne PowerShell oder das Terminal in VS Code im Projekt-Hauptordner.

### 1. Virtuelle Umgebung erstellen

```powershell
py -3.12 -m venv .venv
```

### 2. Virtuelle Umgebung aktivieren

```powershell
.\.venv\Scripts\Activate.ps1
```

Falls die Aktivierung durch die PowerShell-Ausführungsrichtlinie blockiert wird, kann der Interpreter direkt verwendet werden:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Anwendung starten

```powershell
python src/ui_prototype.py
```

Alternativ kann die virtuelle Umgebung direkt verwendet werden:

```powershell
.\.venv\Scripts\python.exe src\ui_prototype.py
```

Die Installation der Python-Abhängigkeiten benötigt normalerweise Internetzugang oder eine vorbereitete Offline-Paketquelle.

Nach erfolgreicher Installation kann die Anwendung lokal ausgeführt werden, ohne eine webbasierte KI-Plattform zu verwenden.

Für eine Installation auf HAW-Labor-PCs ohne Internetzugang müssen die benötigten Abhängigkeiten vorab bereitgestellt oder die Anwendung entsprechend paketiert werden.

## Installation unter Mac

Öffne Terminal oder das Terminal in VS Code im Projekt-Hauptordner.

### 1. Virtuelle Umgebung erstellen

```bash
python3 -m venv .venv
```

### 2. Virtuelle Umgebung aktivieren

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Anwendung starten

```bash
python3 src/ui_prototype.py
```

## Verwendung im Labor

1. Die Anwendung starten.
2. Im Tab **Training** einen Bildordner für `gebrochen` und einen für `intakt` auswählen.
3. Epochen, Batchgröße und Lernrate festlegen.
4. Einen Speicherort für das `.keras`-Modell auswählen.
5. Das Training starten und den Fortschritt beobachten.
6. Im Tab **Test & Ergebnisse** einen unabhängigen Testordner und das zu prüfende Modell auswählen.
7. Die Konfusionsmatrix und Bewertungskennzahlen analysieren.
8. Falls die erforderliche Sitzungs-/Referenzfreigabe vorliegt, die Auswertung als CSV- oder TXT-Bericht exportieren.

## Weitere Python-Module

- `train.py` – CNN-Architektur und Trainingsfunktion für die GUI
- `preprocessing.py` – gemeinsame Bildvorverarbeitung für die Einzelbildklassifikation
- `evaluate.py` – Berechnung der Evaluationskennzahlen und separates Evaluationsskript
- `evaluation_export.py` – Schreiben der CSV- und TXT-Auswertungsberichte
- `inference.py` – Klassifikation eines einzelnen Bildes über die Kommandozeile
- `prepare_dataset.py` – separates Skript zum Aufteilen von Rohdaten in Training, Validierung und Test

## Hinweise und Einschränkungen

- Die Modellqualität hängt von Anzahl, Qualität und Repräsentativität der Bilder ab.
- Unterschiedliche Beleuchtung, Hintergründe, Kameraabstände und Objektorientierungen können die Vorhersagen beeinflussen.
- Eine hohe Accuracy allein belegt nicht die industrielle Eignung eines Qualitätsprüfsystems.
- Trainings- und Testbilder müssen unabhängig voneinander bleiben. Doppelte Bilder in beiden Datensätzen sind zu vermeiden.
- Die Laufzeit hängt von der Hardware und den Trainingsparametern ab.
- TensorFlow 2.21 bietet keine native NVIDIA-GPU-Unterstützung unter Windows. Das Training kann deshalb auf der CPU laufen.

Vor der Nutzung auf HAW-Labor-PCs sollte die Anwendung auf einem sauberen Zielrechner mit den vorgesehenen Benutzerrechten getestet werden.

## Lizenz

Für dieses Projekt ist derzeit keine Lizenz angegeben. Vor einer öffentlichen Weitergabe sollten die Nutzungsrechte für Quellcode, Bilder und weitere Materialien geklärt werden.
