#!/bin/bash

DATA_DIR="project/data/breakhis_no_SOB_M_DC-14-13412_stratified_split_70_15_15_organised_by_mag_256_crops/40X"
SYNTH_DIR="project/synthetic-images/40x-snapshot-008800"
OUT="project/classifier-runs/40x"

# Default
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-default
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-default --synthetic-dir $SYNTH_DIR
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-default --synthetic-dir $SYNTH_DIR --synth-malignant 7000

# Class weights
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-class-weights --class-weights
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-class-weights --synthetic-dir $SYNTH_DIR --class-weights
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-class-weights --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --class-weights

# Dropout 0.5
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-dropout-0.5 --dropout 0.5
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-dropout-0.5 --synthetic-dir $SYNTH_DIR --dropout 0.5
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-dropout-0.5 --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --dropout 0.5

# Frozen backbone
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-frozen --freeze-backbone
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-frozen --synthetic-dir $SYNTH_DIR --freeze-backbone
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-frozen --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --freeze-backbone

# Learning rate 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-lr-0.0001 --lr 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-lr-0.0001 --synthetic-dir $SYNTH_DIR --lr 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-lr-0.0001 --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --lr 0.0001

# Weight decay 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-only-wd-0.0001 --weight-decay 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-0-malignant-wd-0.0001 --synthetic-dir $SYNTH_DIR --weight-decay 0.0001
python project/pipeline/03-classifier-experiments/train_classifier.py --data-dir $DATA_DIR --outdir $OUT/real-and-synthetic-7000-malignant-wd-0.0001 --synthetic-dir $SYNTH_DIR --synth-malignant 7000 --weight-decay 0.0001
