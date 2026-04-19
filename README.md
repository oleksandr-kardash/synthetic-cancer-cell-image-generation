# Synthetic Cancer Cell Image Generation

Final Year Project investigating whether GAN-generated synthetic histopathology images can improve downstream binary classification of breast cancer tissue as benign or malignant. A conditional StyleGAN2-ADA model is trained separately on each BreaKHis magnification level (40x, 100x, 200x, 400x), and the 40x synthetic outputs are used in a series of classifier experiments that isolate the effect of indirect data leakage through the GAN.

The repository is a fork of NVIDIA's [StyleGAN3 codebase](README-stylegan3.md); the original upstream code lives at the repository root (`train.py`, `gen_images.py`, `dnnlib/`, `training/`, etc.) and all project-specific additions live under `project/`.

## Repository structure

```
/                                           # StyleGAN3 upstream (train.py, gen_images.py, dnnlib/, training/, ...)
└── project/
    ├── pipeline/
    │   ├── 01-data-preprocessing/          # scripts to clean, split, and crop the BreaKHis dataset
    │   ├── 02-gan-evaluation/              # FID/KID/precision/recall + Authenticity check
    │   └── 03-classifier-experiments/      # classifier training and the four experiment designs
    ├── reference/                          # PyTorch transfer-learning tutorial (reference code)
    ├── data/                                # raw + preprocessed data (gitignored; see Data setup)
    ├── gan-training-runs/                  # StyleGAN2-ADA training outputs (gitignored)
    ├── synthetic-images/                   # generated synthetic images (gitignored)
    ├── classifier-runs/                    # classifier experiment outputs (gitignored)
    └── gan-evaluation-results/             # GAN evaluation metric and authenticity results
```

## Prerequisites

- Linux system with an NVIDIA GPU (≥12 GB VRAM recommended)
- CUDA toolkit and GCC 7+ for compiling custom ops in `torch_utils/ops/`
- Python 3.11
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

```bash
git clone <repository-url> stylegan3-copy
cd stylegan3-copy
uv venv
source .venv/bin/activate
uv sync
```

All subsequent commands assume the virtual environment is activated and the working directory is the repository root.

## Data setup

Raw datasets and large output artefacts are not tracked in git. Download the following from the project's Google Drive archive and place them at the indicated paths:

```
project/data/BreaKHis_v1/                             # raw BreaKHis dataset (as downloaded from https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
project/data/mkfold/                                  # fold files provided with BreaKHis
project/data/brecahad/                                # BreCaHAD dataset + NVIDIA pickle (required for pipeline validation in step 3)
```

Intermediate files produced by the preprocessing pipeline (sorted image lists, split files, magnification-organised folders, 256x256 crops, training ZIPs) will also be placed under `project/data/` by the scripts below.

## Pipeline

The full pipeline follows five stages. Each stage can be run independently once its inputs are in place.

### 1. Data preprocessing

Run the six preprocessing scripts in order:

```bash
# 1.1 Trim and sort the fold file into a canonical image list (7,909 rows)
python project/pipeline/01-data-preprocessing/01_trim_and_sort_fold.py \
    project/data/mkfold/dsfold1.txt \
    project/data/breakhis_all_images_sorted.txt

# 1.2 Remove the duplicate-mislabelled SOB_M_DC-14-13412 patient (7,786 rows)
python project/pipeline/01-data-preprocessing/02_remove_rows_by_prefix.py \
    project/data/breakhis_all_images_sorted.txt \
    project/data/breakhis_no_SOB_M_DC-14-13412_sorted.txt \
    SOB_M_DC-14-13412

# 1.3 70/15/15 patient-level stratified split
python project/pipeline/01-data-preprocessing/03_patient_split.py \
    project/data/breakhis_no_SOB_M_DC-14-13412_sorted.txt \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt

# 1.4 Copy images into {mag}X/{split}/{class}/ folder structure
python project/pipeline/01-data-preprocessing/04_copy_breakhis_images_into_magnification_split_folders.py \
    project/data/BreaKHis_v1 \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag

# 1.5 Extract 6 x 256x256 overlapping crops per source image
python project/pipeline/01-data-preprocessing/05_extract_256_crops_from_breakhis_images.py \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops

# 1.6 Generate dataset.json files (required for conditional GAN training)
python project/pipeline/01-data-preprocessing/06_generate_dataset_json.py \
    project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops
```

Finally, create a ZIP per magnification using StyleGAN3's `dataset_tool.py` (these are what `train.py` reads):

```bash
# Repeat for each magnification: 40, 100, 200, 400
python dataset_tool.py \
    --source=project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X/train \
    --dest=project/data/zips/breakhis_40x_train_256_crops.zip
```

Optional analysis scripts live under `project/pipeline/01-data-preprocessing/analysis/` and were used during development to investigate patient-ID collisions in the filenames (see the FYP report, §3.2).

### 2. GAN training

Train a conditional StyleGAN2-ADA model for each magnification level, using the configuration from §3.3 of the FYP report:

```bash
# Repeat for each magnification (values shown for 40x)
python train.py \
    --outdir=project/gan-training-runs/40x \
    --cfg=stylegan2 \
    --data=project/data/zips/breakhis_40x_train_256_crops.zip \
    --gpus=1 \
    --batch=16 \
    --gamma=0.8192 \
    --cond=1 \
    --mirror=1 \
    --map-depth=2 \
    --cbase=16384 \
    --aug=ada \
    --target=0.6
```

The training loop saves a network snapshot every 200 kimg and computes FID against the full training set. Stop each magnification once FID has plateaued (see report §4.1.1 for the selected snapshots used in this project).

### 3. GAN evaluation

Two complementary evaluations:

**FID, KID, precision, and recall** for the selected snapshot of each magnification and the BreCaHAD NVIDIA reference pickle:

```bash
bash project/pipeline/02-gan-evaluation/gan_calculate_evaluation_metrics.sh
```

Results are appended to `project/gan-evaluation-results/gan_evaluation_metrics.txt`.

**Authenticity check** (Alaa et al., 2022) — per-sample memorisation test against the training set:

```bash
bash project/pipeline/02-gan-evaluation/run_authenticity_check.sh
```

Results are saved to `project/gan-evaluation-results/authenticity-check/{mag}-snapshot-{id}/authenticity_results.txt`.

### 4. Synthetic image generation

Generate synthetic images for the downstream classifier experiments. The project uses only the 40x generator:

```bash
# Benign (class=0), 12,001 images (seeds 0-12000)
python gen_images.py \
    --outdir=project/synthetic-images/40x-snapshot-008800/benign \
    --seeds=0-12000 \
    --class=0 \
    --network=project/gan-training-runs/40x/00005-stylegan2-breakhis_40x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-008800.pkl

# Malignant (class=1), 10,001 images (seeds 0-10000)
python gen_images.py \
    --outdir=project/synthetic-images/40x-snapshot-008800/malignant \
    --seeds=0-10000 \
    --class=1 \
    --network=project/gan-training-runs/40x/00005-stylegan2-breakhis_40x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-008800.pkl
```

### 5. Classifier experiments

Four experimental designs of increasing rigour, each in its own subfolder. Run them all in sequence:

```bash
bash project/pipeline/03-classifier-experiments/run_all_experiments.sh
```

Or run them individually:

```bash
# 5.1 70/15/15 baseline with six hyperparameter configurations
bash project/pipeline/03-classifier-experiments/01-baseline-70-15-15/run_baseline_experiments.sh

# 5.2 40/20/40 split with two seeds
bash project/pipeline/03-classifier-experiments/02-split-40-20-40/run_40_20_40_experiments.sh

# 5.3 5-fold patient-level stratified cross-validation
bash project/pipeline/03-classifier-experiments/03-cross-validation/run_cv_experiments.sh

# 5.4 Controlled A/B leakage experiment
bash project/pipeline/03-classifier-experiments/04-gan-leakage/run_gan_leakage_experiments.sh
```

All classifier runs use the shared `train_classifier.py` under `project/pipeline/03-classifier-experiments/`. Outputs (`results.json`, per-class metrics, training logs) are written to `project/classifier-runs/40x/<experiment-name>/`.

## Results

- **GAN evaluation metrics** (FID, KID, precision, recall for all four magnifications + BreCaHAD validation): `project/gan-evaluation-results/gan_evaluation_metrics.txt`
- **Authenticity check results**: `project/gan-evaluation-results/authenticity-check/`
- **Classifier experiment results**: `project/classifier-runs/` (gitignored)

## Report

The full FYP report discussing methodology, results, and the indirect data leakage finding is kept separately on Google Drive alongside the raw data and training artefacts.

## Credits

This project builds on NVIDIA's StyleGAN3 codebase. The original upstream README and license are preserved as [`README-stylegan3.md`](README-stylegan3.md) and [`LICENSE.txt`](LICENSE.txt).
