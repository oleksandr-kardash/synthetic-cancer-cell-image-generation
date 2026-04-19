#!/bin/bash

DATA_DIR="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X"
SYNTH_DIR="project/synthetic-images/40x-snapshot-008800"
OUT="project/classifier-runs/40x"

# Seed 42
python project/pipeline/03-classifier-experiments/02-split-40-20-40/40_20_40_synthetic_augmentation_experiment.py --outdir $OUT/40-20-40-synthetic-augmentation-experiment-0-malignant-seed42 --data-dir $DATA_DIR --synthetic-dir $SYNTH_DIR --seed 42
python project/pipeline/03-classifier-experiments/02-split-40-20-40/40_20_40_synthetic_augmentation_experiment.py --outdir $OUT/40-20-40-synthetic-augmentation-experiment-7000-malignant-seed42 --data-dir $DATA_DIR --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --seed 42

# Seed 123
python project/pipeline/03-classifier-experiments/02-split-40-20-40/40_20_40_synthetic_augmentation_experiment.py --outdir $OUT/40-20-40-synthetic-augmentation-experiment-0-malignant-seed123 --data-dir $DATA_DIR --synthetic-dir $SYNTH_DIR --seed 123
python project/pipeline/03-classifier-experiments/02-split-40-20-40/40_20_40_synthetic_augmentation_experiment.py --outdir $OUT/40-20-40-synthetic-augmentation-experiment-7000-malignant-seed123 --data-dir $DATA_DIR --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --seed 123
