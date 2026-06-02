![FluoroBoxer logo](assets/fluoroboxer-logo.svg)

[![Python Compatibility](https://github.com/LSUHSC-CAIPP-Core-B/FluoroBoxer/actions/workflows/python-compat.yml/badge.svg)](https://github.com/LSUHSC-CAIPP-Core-B/FluoroBoxer/actions/workflows/python-compat.yml)

FluoroBoxer is a reproducible annotation pipeline for detecting dead cells in phase-contrast microscopy images.
It uses SYTOX Green fluorescence as a reference channel to generate bounding boxes for model training, minimizing manual labeling effort while preserving annotation quality.

## Why FluoroBoxer?

Manual dead-cell annotation is slow, subjective, and difficult to scale. FluoroBoxer helps by:

- automating box generation from fluorescence-guided masks,
- providing an interactive parameter tuning app (Streamlit),
- supporting common training formats (YOLO, Pascal VOC XML, TFRecord),
- supporting augmentation and train/validation/test splitting workflows.

---

## Repository Layout

```text
FluoroBoxer/
├── CellProcessor/                    # Core reusable image/label processing functions
├── cell_image_projection.py          # Streamlit app for parameter tuning and visual QA
├── annotate_cell_images.ipynb        # Main automatic annotation workflow (YOLO labels)
├── prepare_background_images.ipynb   # Background/negative sample extraction
├── augment_dataset.ipynb             # Augment images and labels
├── train_validation_test_split.ipynb # Build train/val/test partitions
├── convert_yolo_to_pascal_voc.ipynb  # YOLO → Pascal VOC conversion
├── convert_xml_to_tfrecord.py        # Pascal VOC XML → TFRecord converter
├── verify_yolo_bounding_boxes.ipynb  # Visual label verification
├── dataset.csv                       # Dataset registry for Streamlit app
├── labels.pbtxt                      # Default label map for TFRecord export
├── params.csv                        # Saved preprocessing parameter registry
└── Data/                             # Local dataset storage (raw, intermediate, exports)
```

---

## Data Organization

Expected data layout:

```text
Data/
├── <DeathType>/
│   ├── <CellType>_Phase/
│   ├── <CellType>_Green/
│   ├── <CellType>_Phase_crop/
│   ├── <CellType>_Green_crop/
│   ├── <CellType>_Labeled_phase/
│   ├── <CellType>_Masks_phase/
│   └── <CellType>_*_aug/
└── final_Data_set/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── xml_outputs/
    └── TFRecord_output/
```

---

## Installation

```bash
python3.13 -m venv .venv
# or: python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

FluoroBoxer is currently validated on Python 3.8 and Python 3.13 for the core app and tests.

### Compatibility Matrix

- Python 3.8: supported for the core workflow and test suite
- Python 3.13: supported for the core workflow and test suite
- Continuous verification: `.github/workflows/python-compat.yml` runs the test suite on both versions

### Optional TFRecord export dependencies

The TFRecord converter depends on TensorFlow but no longer requires the TensorFlow Object Detection API.
The exporter is validated on both supported Python versions, but the TensorFlow stack is version-specific:

- Python 3.8 installs `tensorflow==2.13.1`
- Python 3.13 installs `tensorflow==2.20.0`

Create a dedicated environment for TFRecord export:

```bash
python3.13 -m venv .venv-tfrecord
# or: python3.8 -m venv .venv-tfrecord
source .venv-tfrecord/bin/activate
pip install --upgrade pip
pip install -r requirements-tfrecord.txt
```

`labels.pbtxt` is included in the repository for the default single-class export path used by the bundled XML examples.

### Key runtime dependencies

- Python 3.8 or Python 3.13 for the core workflow
- OpenCV, NumPy, Pandas
- Streamlit
- TensorFlow (optional, for TFRecord conversion)

---

## Quick Start

### 1) Tune preprocessing parameters interactively

```bash
streamlit run cell_image_projection.py
```

Use the sidebar to select a dataset, navigate images, and tune parameters. Save stable settings to `params.csv`.

### 2) Generate annotations

Run notebook:

- `annotate_cell_images.ipynb`

### 3) Build negatives/background patches

Run notebook:

- `prepare_background_images.ipynb`

### 4) Augment data

Run notebook:

- `augment_dataset.ipynb`

### 5) Split into train/val/test

Run notebook:

- `train_validation_test_split.ipynb`

### 6) Export to other formats

- YOLO → Pascal VOC: `convert_yolo_to_pascal_voc.ipynb`
- Pascal VOC XML → TFRecord:

```bash
python convert_xml_to_tfrecord.py \
  --xml_dir ./Data/final_Data_set/xml_outputs \
  --labels_path ./labels.pbtxt \
  --output_path ./Data/final_Data_set/TFRecord_output/dataset.record \
  --image_dir ./Data/final_Data_set/images
```

If TensorFlow export dependencies are missing, the script exits with guidance to install `requirements-tfrecord.txt`.

### 7) Verify labels

Run notebook:

- `verify_yolo_bounding_boxes.ipynb`

---

## Reproducibility Notes

- `dataset.csv` controls dataset discovery in the Streamlit app.
- `params.csv` stores preprocessing settings and metadata.
- Keep these two files versioned to ensure reproducible processing.

---

## Citation

If FluoroBoxer contributes to your work, please cite the repository and associated publication.

Repository: https://github.com/LSUHSC-CAIPP-Core-B/FluoroBoxer

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
