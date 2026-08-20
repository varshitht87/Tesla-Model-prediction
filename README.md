# Tesla Car Model Classifier

Transfer-learning image classifier that sorts photos of Tesla cars into four
model classes (**Model E, S, X, Y**) using pretrained CNN backbones.

**Author:** Sai Krishna Varshith 

## Dataset

- **Source:** [Tesla Car Models Dataset](https://www.kaggle.com/datasets/meeratif/tesla-car-dataset) on Kaggle
  (also viewable via the notebook that uses it: [kaggle.com/code/meeratif/tesla-car-models-dataset/input](https://www.kaggle.com/code/meeratif/tesla-car-models-dataset/input))
- 1,311 images total — 1,049 for training, 262 for validation (80/20 split)
- 4 classes, imbalanced (Model_S and Model_X underrepresented) — handled with
  computed class weights
- Images resized to 224×224, loaded via `tf.keras.utils.image_dataset_from_directory`

## Repo structure

```
tesla-model-classifier/
├── README.md
├── notebooks/
│   ├── 01_baseline_four_models.ipynb       # graded CA: 4 backbones compared
│   └── 02_custom_head_experiments.ipynb    # follow-up: custom heads on EfficientNetB0
└── assets/
    └── (screenshots referenced from notebook 1's analysis markdown)
```

## Notebook 1 — Baseline: four pretrained backbones

Each of **EfficientNetB0, ResNet50, MobileNetV2, DenseNet121** is trained in two phases:

1. **Phase 1 (feature extraction):** base model frozen, only a new
   classification head trained (10 epochs, lr = 1e-4).
2. **Phase 2 (fine-tuning):** last 50 layers of the base model unfrozen
   (BatchNorm layers kept frozen), trained with a lower learning rate
   (3e-5), label smoothing (0.1), `ReduceLROnPlateau`, and early stopping
   on macro F1.

### Results (best validation scores, Phase 2)

| Model          | Params | Val Accuracy | Val F1 (macro) | Train/Val Gap |
|----------------|-------:|-------------:|----------------:|---------------:|
| **MobileNetV2**| 3.4M   | **67.6%**    | **0.675**        | Moderate       |
| ResNet50       | 25.6M  | 65.3%        | —                | Largest (31.8%) — overfits |
| EfficientNetB0 | 5.3M   | 61.1%        | 0.688*           | Moderate       |
| DenseNet121    | 8.0M   | 57.6%        | —                | Smallest (6.9%) — underfits |

\* EfficientNetB0's val F1 (0.688) is the model's actual best epoch, which
doesn't coincide with its best val-accuracy epoch — see notebook 1 for the
full per-epoch curves and analysis.

**Best overall: MobileNetV2** — highest val accuracy and F1 despite being
the smallest model, with a moderate (not worst) overfitting gap.

**Why no model reached 90%:** dataset is small (1,049 training images) for
these model sizes, the four Tesla models share a similar minimalist design
language, and ImageNet features aren't car-specific. Full discussion is in
notebook 1's closing summary cell.

## Notebook 2 — Extended: custom classification heads (EfficientNetB0)

Follow-up experiments testing whether a heavier classification head (extra
Dense layers, BatchNorm, dropout) on top of EfficientNetB0 beats the plain
GAP → Dropout → Dense head from notebook 1.

Three runs, built through one shared `build_efficientnet_custom_head()` /
`train_custom_head()` function pair rather than copy-pasted per run:

- **Run A** — 256→64 head, fine-tuned base (last 50 layers unfrozen)
- **Run B** — 512→128 head, same fine-tuned base
- **Run C** — 256→64 head, base fully frozen, higher learning rate (1e-3)

This notebook also documents (in its "Debugging notes" cell) six bugs found
in the original exploratory version — overwritten training history,
overwritten checkpoint files, a run that silently reverted its fine-tuning,
a metric being monitored that wasn't actually being tracked, and a dropped
`class_weight` — all fixed in the version here.

**Status:** results table in notebook 2 is a template — the three runs need
to be executed (GPU + Drive-mounted dataset) to populate real numbers.

## How to run

Both notebooks are built for Google Colab with a T4 GPU:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/meeratif/tesla-car-dataset)
   and upload it to `MyDrive/Deep Learning/Tesla Models/` in Google Drive,
   with one subfolder per class (`Model E/`, `Model S/`, `Model X/`, `Model Y/`)
2. Open a notebook in Colab, mount Google Drive when prompted
3. Run notebook 1 top to bottom for the baseline comparison
4. Run notebook 2 top to bottom (independently — it reloads its own data)
   for the custom-head experiments
