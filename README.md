# UNet++ + QCNN Pipeline for PBS Leukemia Dataset

This project combines a U-Net++ segmentation model and a quantum-classical CNN classifier for blood-cell image analysis using the PBS dataset.

## Project structure

- `train_unetpp.py` — train the segmentation model
- `unetpp.py` — UNet++ implementation
- `dataset.py` — custom segmentation dataset
- `utils.py` — mask generation utilities
- `train_qcnn.py` — QCNN training workflow
- `train_qcnn_cells.py` — cell-level QCNN training workflow
- `classify_image.py` and `classify_image_cells.py` — inference scripts
- `inference_unetpp.py` — segmentation inference on new images

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the dataset in the expected folder structure:

```text
data/
  images/
  masks/
```

## Training

Run the segmentation model:

```bash
python train_unetpp.py
```

Run the QCNN classifier:

```bash
python train_qcnn.py
```

## Notes

- Model weights and generated outputs are intentionally excluded from version control using `.gitignore`.
- The segmentation mask naming should remain consistent with the image filenames or the dataset loader should resolve the correct mask path.
- This repository is intended for research and experimentation, not production deployment.
