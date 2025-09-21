
# Brain Tumor MRI Image Classification and Segmentation

**[Read the full paper (PDF)](./paper.pdf)**

Multi-task deep learning pipeline for the BRISC 2025 brain tumor MRI dataset. This project implements and compares:

1. U-Net (Segmentation)
2. Attention U-Net (Segmentation)
3. U-Net-based Classifier (Classification)
4. Joint U-Net (Simultaneous Segmentation + Classification)

It evaluates Dice, IoU, Pixel Accuracy for segmentation and Accuracy for classification. Visualization utilities are included for predictions and comparative metric plots.

---
## Key Features
- Custom `DataGenerator` for segmentation, classification, and joint tasks
- Multiple U-Net variants (standard, attention, and joint multi-head)
- Mixed loss functions: BCE + Dice for segmentation, categorical crossentropy for classification
- Data augmentation (flip, rotation, brightness, contrast)
- Training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Comparison plots and metric export (JSON)
- Visualization of overlays and sample predictions

---

## Project Structure
```
brain-tumor-mri-image-classification-and-sgmentation/
├── main.ipynb         # End-to-end training & evaluation notebook
├── paper.pdf          # Compiled paper (if present)
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignore rules
├── LICENSE            # MIT License
├── README.md          # Project documentation
├── references.bib     # BibTeX references for paper
└── figures/           # Exported result figures (PNG)
```

> Note: Raw dataset files and trained model weight files (`*.h5`) are intentionally excluded from version control. Only result figures are included in `figures/`.

---

## Dataset
This code expects the BRISC 2025 dataset with two tasks:
- Segmentation: `segmentation_task/train/{images,masks}` and `segmentation_task/test/{images,masks}`
- Classification: `classification_task/train/<class_name>/` and `classification_task/test/<class_name>/`

Classes (adjust if different):
```
['glioma', 'meningioma', 'no_tumor', 'pituitary']
```

> Update dataset base paths in `main.ipynb` (`Config.SEG_BASE_PATH` and `Config.CLS_BASE_PATH`) to match your local setup.

---

## Environment Setup
1. (Optional) Create a virtual environment:
  ```bash
  python -m venv .venv
  # On Windows:
  .venv\Scripts\activate
  # On Linux/macOS:
  source .venv/bin/activate
  ```
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. (Optional) For GPU: Install matching TensorFlow and CUDA/cuDNN as per official TensorFlow instructions.

---

## Running the Pipeline
Open `main.ipynb` and run all cells in order. The notebook will:
1. Load dataset paths
2. Build data generators
3. Train each model variant (saving best weights, not tracked by git)
4. Evaluate models and export metrics to `model_metrics.json`
5. Produce comparative bar charts and sample prediction overlays (saved in `figures/`)

> If running outside Kaggle, adjust load/save paths in the notebook as needed.

---

## Outputs (Not Tracked by Git)

> If you wish to version large files (e.g., model weights), use Git LFS.

## Related Paper

### Title
Brain Tumor MRI Image Classification and Segmentation Using Deep Learning

### Authors
*Author names as listed in the paper*

### Abstract
We present a reproducible multi-task deep learning
approach for brain tumor MRI that jointly performs binary
tumor segmentation and four-class subtype classification on the
BRISC 2025 dataset. We evaluate a baseline U-Net, an attention-
augmented variant, a standalone classifier, and a shared- encoder
joint model, reporting Dice, IoU, pixel accuracy, and classification
F1/accuracy metrics. Attention consistently refines lesion bound-
aries; the joint model maintains strong classification performance
with a modest segmentation trade-off, offering an efficient single-
checkpoint solution for deployment.

### Paper Access
The full paper is included in this repository as [`paper.pdf`](./paper.pdf). Please refer to it for a comprehensive description of the methodology, experiments, and results.

---
## Customization
- Adjust image size and hyperparameters in the `Config` class.
- Modify augmentation logic in `DataGenerator.augment_data`.
- Change classification head depth via `classifier_type` ("simple" or "deep").
- Tune loss weighting in `train_joint_model` via `seg_weight` and `class_weight`.

---
## Potential Improvements
- Add mixed precision training for speed (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`)
- Implement k-fold cross-validation
- Add test-time augmentation (TTA) for segmentation
- Replace backbone with pretrained encoder (e.g., EfficientNet) via transfer learning
- Incorporate uncertainty estimation or ensemble averaging

---
## Reproducibility Notes
Random seeds are set for `numpy`, `tensorflow`, and Python `random`, but full determinism may still vary across hardware/backends.

---
## Citation
If you use this repository in academic work, please cite the BRISC 2025 dataset (add official citation once available) and this repository:
```
@misc{your_repo_2025,
  title  = {Brain Tumor MRI Image Classification and Segmentation},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/<your-username>/brain-tumor-mri-image-classification-and-sgmentation}
}
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
## Disclaimer
This project is for research and educational purposes only and is **not** a medical device. Predictions must not be used for clinical decisions without proper validation and regulatory approval.

---
## Contributing
Pull requests welcome. For larger changes, open an issue first to discuss proposed modifications.

---
## Contact
Questions / suggestions: open an issue or reach out via GitHub.
