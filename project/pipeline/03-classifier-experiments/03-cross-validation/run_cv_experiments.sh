#!/bin/bash

DATA_DIR="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X"
SPLIT_FILE="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15.txt"
SYNTH_DIR="project/synthetic-images/40x-snapshot-008800"
OUT="project/classifier-runs/40x"

# Seed 42
python project/pipeline/03-classifier-experiments/03-cross-validation/cross_validation.py --outdir $OUT --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 42
python project/pipeline/03-classifier-experiments/03-cross-validation/cross_validation.py --outdir $OUT --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 42 --synth-malignant 7000

# Seed 123
python project/pipeline/03-classifier-experiments/03-cross-validation/cross_validation.py --outdir $OUT --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 123
python project/pipeline/03-classifier-experiments/03-cross-validation/cross_validation.py --outdir $OUT --data-dir $DATA_DIR --split-file $SPLIT_FILE --synthetic-dir $SYNTH_DIR --seed 123 --synth-malignant 7000
