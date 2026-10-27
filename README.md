# Handwritten Detection

AI-based Devanagari handwritten character recognition using ResNet50, DenseNet121, a Vision Transformer, and a weighted soft-voting ensemble.

The application classifies uploaded handwritten character images into the dataset classes discovered from the existing `DevanagariHandwrittenCharacterDataset` in this project.

## Project Overview

Handwritten Detection is an end-to-end research system that:

1. Discovers the existing Devanagari dataset already in the project folder.
2. Uses the original `Train` folder for an 85% / 15% stratified train/validation split.
3. Uses the original `Test` folder only as a final independent test set.
4. Trains ResNet50, DenseNet121, and ViT-B/16 with transfer learning.
5. Searches ensemble weights on validation predictions.
6. Evaluates accuracy, precision, recall, F1, One-vs-Rest ROC-AUC, confusion matrices, and error analysis.
7. Produces Grad-CAM, ViT attention, and t-SNE visualizations.
8. Serves a Streamlit app named **Handwritten Detection**.

No experimental accuracy numbers are stored in this README. Metrics appear only after you train and evaluate the models.

## Dataset

This project uses the **Devanagari Handwritten Character Dataset** that is already present in the repository. Do not download a replacement dataset and do not generate a new one.

The current copy lives at:

```text
archive/DevanagariHandwrittenCharacterDataset/
├── Train/
└── Test/
```

The code also accepts the dataset at the project root:

```text
DevanagariHandwrittenCharacterDataset/Train
DevanagariHandwrittenCharacterDataset/Test
```

or through the environment variable `DEVANAGARI_DATASET_PATH`.

Class names are read from the folder names. The local dataset contains 46 classes (36 Devanagari characters and 10 digits), 78,200 training images, and 13,800 test images in PNG format.

## Dataset Structure

```text
DevanagariHandwrittenCharacterDataset/
│
├── Train/
│   ├── character_1_ka/
│   ├── character_2_kha/
│   ├── ...
│   └── digit_9/
│
└── Test/
    ├── character_1_ka/
    ├── character_2_kha/
    ├── ...
    └── digit_9/
```

Split protocol:

| Split | Source | Ratio |
| --- | --- | --- |
| Training | original `Train` | 85%, stratified |
| Validation | original `Train` | 15%, stratified |
| Testing | original `Test` | 100%, untouched |

## Installation

```bash
cd hindi-handwritten
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch CUDA wheels should match your local GPU driver. CPU training is supported automatically.

## Environment Setup

Configuration lives in `config.py`:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_CLASSES = 46   # overwritten by the number of discovered folders
VAL_SPLIT = 0.15
RANDOM_SEED = 42
```

Device selection:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Reproducibility seed `42` is applied to Python, NumPy, PyTorch, and CUDA.

## Training

```bash
python train.py
```

Optional arguments:

```bash
python train.py --epochs 10 --models resnet50 densenet121 vit
```

`train.py` will:

- discover the existing dataset
- save class names to `checkpoints/class_names.json`
- train each requested model
- write `checkpoints/best_resnet50.pth`, `checkpoints/best_densenet121.pth`, and `checkpoints/best_vit.pth`
- search ensemble weights on the validation set
- save `checkpoints/ensemble_weights.json` and `outputs/results/training_history.json`
- write dataset and training plots under `outputs/plots/`

Early stopping uses validation accuracy with patience 5. Mixed precision is enabled when CUDA is available.

## Evaluation

```bash
python evaluate.py
```

This script evaluates the independent `Test` split only and writes:

- `outputs/results/evaluation_results.json`
- confusion matrices
- One-vs-Rest ROC and precision-recall curves
- misclassified examples
- ablation-study table
- Grad-CAM / ViT attention samples
- t-SNE feature visualization when possible

If training has not been run, evaluation and the Streamlit dashboards show:

```text
Not Available — Run Training
```

They do not invent accuracy values.

## Running Streamlit

```bash
streamlit run app/streamlit_app.py
```

The UI title is **Handwritten Detection**, with the subtitle *AI-Based Devanagari Handwritten Character Recognition*.

Sidebar pages:

- Home
- Character Prediction
- Model Performance
- Dataset Analysis
- Explainable AI
- About Project

Upload PNG, JPG, or JPEG handwritten character images on the prediction page. The app loads checkpoints with `@st.cache_resource` and does not retrain on startup.

Single-image CLI inference:

```bash
python inference.py path\to\character.png
python ensemble_predict.py path\to\character.png
```

## Research Outputs

After training and evaluation the following artifacts are produced:

| Output | Location |
| --- | --- |
| Table 1 Dataset distribution | `outputs/results/dataset_stats.json` |
| Table 3 Individual model performance | `outputs/results/table3_model_performance.csv` |
| Table 5 Ablation study | `outputs/results/table5_ablation_study.csv` |
| Figure 1 Architecture | `outputs/plots/figure1_system_architecture.png` |
| Figure 2 Dataset distribution | `outputs/plots/figure2_dataset_distribution.png` |
| Figure 3 Training curves | `outputs/plots/<model>_loss.png` and `_accuracy.png` |
| Figure 4 Confusion matrix | `outputs/confusion_matrices/` |
| Figure 5 ROC curves | `outputs/plots/` |
| Figure 6 Grad-CAM | `outputs/plots/figure6_gradcam_*.png` |
| Figure 7 ViT attention | `outputs/plots/figure7_vit_attention.png` |
| Figure 8 t-SNE | `outputs/plots/figure8_tsne_features.png` |

## Project Layout

```text
hindi-handwritten/
├── archive/DevanagariHandwrittenCharacterDataset/
├── dataset/
├── preprocessing/
├── models/
├── training/
├── ensemble/
├── evaluation/
├── explainability/
├── app/streamlit_app.py
├── checkpoints/
├── outputs/
├── notebooks/
├── config.py
├── train.py
├── evaluate.py
├── inference.py
└── ensemble_predict.py
```

## License and academic use

This repository is intended for academic demonstration of Devanagari handwritten character recognition. Keep the original dataset files unchanged.
