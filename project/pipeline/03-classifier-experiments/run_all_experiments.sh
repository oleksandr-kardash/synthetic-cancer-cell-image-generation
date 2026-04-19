#!/bin/bash

# Run all classifier experiments in narrative order.
# 1. 70/15/15 baseline + hyperparameter tuning
# 2. 40/20/40 synthetic augmentation experiment
# 3. 5-fold cross-validation
# 4. GAN leakage experiment

bash project/pipeline/03-classifier-experiments/01-baseline-70-15-15/run_baseline_experiments.sh
bash project/pipeline/03-classifier-experiments/02-split-40-20-40/run_40_20_40_experiments.sh
bash project/pipeline/03-classifier-experiments/03-cross-validation/run_cv_experiments.sh
bash project/pipeline/03-classifier-experiments/04-gan-leakage/run_gan_leakage_experiments.sh
