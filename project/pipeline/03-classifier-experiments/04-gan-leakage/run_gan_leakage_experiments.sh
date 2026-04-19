#!/bin/bash

DATA_DIR="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X"
SPLIT_FILE="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt"
SYNTH_DIR="project/synthetic-images/40x-snapshot-008800"
OUT="project/classifier-runs/40x"

# Seed 42
python project/pipeline/03-classifier-experiments/04-gan-leakage/gan_leakage_experiment.py --outdir $OUT/gan-leakage-experiment-0-malignant-seed42 --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 42
python project/pipeline/03-classifier-experiments/04-gan-leakage/gan_leakage_experiment.py --outdir $OUT/gan-leakage-experiment-7000-malignant-seed42 --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --seed 42

# Seed 123
python project/pipeline/03-classifier-experiments/04-gan-leakage/gan_leakage_experiment.py --outdir $OUT/gan-leakage-experiment-0-malignant-seed123 --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 123
python project/pipeline/03-classifier-experiments/04-gan-leakage/gan_leakage_experiment.py --outdir $OUT/gan-leakage-experiment-7000-malignant-seed123 --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --seed 123
