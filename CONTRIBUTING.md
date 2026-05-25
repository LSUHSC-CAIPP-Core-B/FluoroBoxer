# Contributing to FluoroBoxer

Thanks for your interest in improving FluoroBoxer.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Contribution Scope

Useful contributions include:

- image processing improvements in `CellProcessor/`,
- notebook reproducibility and documentation updates,
- bug fixes in Streamlit UX and data I/O,
- benchmark and evaluation improvements,
- test coverage for reusable Python modules.

## Coding Guidelines

- Keep functions small and focused.
- Add/maintain docstrings for non-trivial behavior.
- Avoid breaking file and column naming conventions used by notebooks.
- Prefer explicit paths and robust error messages for data-loading code.

## Testing

Run checks before submitting:

```bash
python -m compileall CellProcessor cell_image_projection.py convert_xml_to_tfrecord.py
python -m pytest
```

If TensorFlow Object Detection API is unavailable in your environment, note it in your PR and include results for all checks that can run.

## Pull Requests

Please include:

1. What changed and why.
2. Reproduction/validation steps.
3. Any data format or schema impact.
4. Screenshots for visible Streamlit UI changes.

## Release Checklist

Before tagging a release:

1. Confirm notebooks run top-to-bottom in a supported Python environment with current dependency versions.
2. Validate at least one end-to-end dataset pass (annotation -> augmentation -> split).
3. Verify class mappings for any label remapping tasks.
4. Confirm sample visual QA in `verify_yolo_bounding_boxes.ipynb`.
5. Update README and changelog/release notes.

## Data and Privacy

- Do not commit proprietary microscopy data without approval.
- Avoid adding large raw data files unless required for a reproducible example.
- Remove personally identifying metadata from shared datasets when applicable.
