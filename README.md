# FluoroBoxer

An automated pipeline for annotating dead cells in phase-contrast microscopy images. FluoroBoxer uses SYTOX Green fluorescence as a reference signal to generate high-quality bounding-box annotations in YOLO format, enabling the development of deep learning models for dead-cell detection and automated image analysis.


## Overview

FluoroBoxer addresses the challenge of accurately annotating dead cells in phase-contrast microscopy images by automating the annotation pipeline:

- **Reference-guided detection**: Uses SYTOX Green fluorescence signal as ground truth to identify dead cells
- **Parameter optimization**: Interactive "Cell Image Projector" tool for fine-tuning preprocessing parameters
- **Automated annotation**: Generates high-quality bounding boxes with minimal manual intervention
- **Scalable processing**: Batch processes images and crops them into 128×128-pixel patches with configurable overlap
- **Multiple output formats**: Exports annotations in YOLO, Pascal VOC, and TFRecord formats
- **Data augmentation**: Applies brightness/contrast variations while maintaining annotation integrity

Designed for high-throughput microscopy image analysis where manual annotation is time-consuming and error-prone.

## Project Structure

### Core Components

#### 1. **Cell Image Projector** (Parameter Optimization)
- **`cell_image_projection.py`** (Streamlit App)  
  Interactive web-based tool for optimizing preprocessing parameters that accurately delineate dead-cell boundaries in phase-contrast images. Real-time visualization shows the effect of each parameter (brightness, contrast, thresholding, erosion, dilation) on cell detection, with reference to SYTOX Green fluorescence. Saves optimized parameter sets to `params.csv` for reproducible batch processing.
  
  **Features:**
  - Live preview of preprocessing stages
  - Adjustable brightness, contrast, threshold, erosion, and dilation parameters
  - Bounding box overlay and full-screen image viewing
  - Parameter persistence and history tracking

#### 2. **Automated Annotation**
- **`annotate_cell_images.ipynb`**  
  Batch process phase-contrast images with optimized FluoroBoxer parameters to automatically generate bounding boxes around dead cells. Crops raw images into 128×128-pixel patches with configurable overlap to ensure complete image coverage. Outputs YOLO-formatted labels and image patches ready for model training.

- **`prepare_background_images.ipynb`**  
  Extract background patches containing no dead cells using the same parameter set. Provides negative samples for balanced training datasets.

#### 3. **Data Augmentation**
- **`augment_dataset.ipynb`**  
  Apply brightness and contrast augmentations to images while automatically updating bounding box annotations. Uses Albumentations library to maintain data-label consistency throughout transformations.

#### 4. **Dataset Organization**
- **`train_validation_test_split.ipynb`**  
  Split augmented dataset into train/validation/test partitions (typical 60/20/20 split).

#### 5. **Format Conversion**
- **`convert_yolo_to_pascal_voc.ipynb`**  
  Convert YOLO bounding box format to Pascal VOC XML format.

- **`convert_xml_to_tfrecord.py`** (CLI Tool)  
  Convert Pascal VOC XML annotations to TensorFlow Record format for TF 1.x object detection models.
  ```bash
  python convert_xml_to_tfrecord.py -x ./annotations -l labels.pbtxt -o dataset.record -i ./images
  ```

- **`remap_yolo_class_labels.ipynb`**  
  Reassign class IDs in YOLO annotations (useful for label consolidation or remapping).

#### 6. **Validation & Visualization**
- **`verify_yolo_bounding_boxes.ipynb`**  
  Visually verify bounding box accuracy by overlaying boxes on images.

#### 7. **Utility Module**
- **`CellProcessor/`** (Package)  
  Reusable functions for image processing, bounding box generation, label I/O, and visualization. Core operations used throughout the pipeline.

## Data Directory Structure

```
Data/
├── Necroptosis/           # Cell death type directory (or other death pathways)
│   ├── MEF_Phase/         # Phase-contrast images (reference channel)
│   ├── MEF_Green/         # SYTOX Green fluorescence images (reference signal)
│   ├── MEF_Phase_crop/    # Cropped Phase-contrast images(128×128)
│   ├── MEF_Green_crop/    # Cropped fluorescence images (128×128)
│   ├── MEF_Labeled_phase/ # Annotated phase-contrast images (128×128)
│   ├── MEF_Masks_phase/   # Segmentation masks from fluorescence (128×128)
│   └── MEF_*_aug/         # Augmented image patches (128×128)
└── final_Data_set/        # Processed dataset
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/           # YOLO format annotations
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── xml_outputs/      # Pascal VOC XML annotations
    └── TFRecord_output/  # TFRecord files
```

## Configuration

### Dataset Registry (`dataset.csv`)
Defines available datasets and their locations for the preprocessing pipeline:
- **ID**: Unique identifier for the dataset
- **Cell_type**: Cell line name (e.g., MEF - Mouse Embryonic Fibroblasts)
- **Death_type**: Type of cell death being studied (e.g., Necroptosis, Apoptosis, ...)
- **Image_path**: Root directory path containing the raw images
- **Description**: Brief description or notes about the dataset

Used by `cell_image_projection.py` to populate the dataset dropdown and locate image files.

### Parameters File (`params.csv`)
Stores preprocessing parameter configurations with metadata:
- **ID**: Unique parameter configuration identifier
- **Brightness, Contrast, Threshold, Erosion, Dilation**: Preprocessing parameter values
- **Time**: Timestamp of when the configuration was saved
- **Description**: User notes for the configuration

Enables parameter reproducibility and experimentation tracking across preprocessing sessions.


## Quick Start

### 1. Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Image Preprocessing
```bash
streamlit run cell_image_projection.py
```
Opens browser at `localhost:8501`. Select dataset, adjust parameters, and save optimal settings.

### 3. Annotate Images
Run `annotate_cell_images.ipynb` to generate YOLO-format bounding box labels.

### 4. Augment Dataset
Run `augment_dataset.ipynb` to apply brightness/contrast variations while preserving annotations.

### 5. Split Dataset
Run `train_validation_test_split.ipynb` to create train/val/test partitions.

### 6. Convert Formats (if needed)
- **To Pascal VOC:** Run `convert_yolo_to_pascal_voc.ipynb`
- **To TFRecord:** Use `convert_xml_to_tfrecord.py`

### 7. Verify Results
Run `verify_yolo_bounding_boxes.ipynb` to visualize annotated images.

## Workflow

```
Raw Images
    ↓
[Preprocessing] → Parameter tuning in cell_image_projection.py
    ↓
[Annotation] → YOLO labels in annotate_cell_images.ipynb
    ↓
[Augmentation] → Brightness/contrast variations in augment_dataset.ipynb
    ↓
[Train/Val/Test Split] → train_validation_test_split.ipynb
    ↓
[Format Conversion] → XML or TFRecord as needed
    ↓
Ready for Model Training
```


If you use FluoroBoxer in your research, please cite the corresponding paper and GitHub repository:


**GitHub:** https://github.com/LSUHSC-CAIPP-Core-B/FluoroBoxer

## *Reproducible Image Analysis**: Parameter tracking enables consistent, reproducible analysis across experiments

## License

GNU General Public License v3.0 - See [LICENSE](LICENSE) for details.
