#!/bin/bash
# Calculate FID, KID, precision, and recall for all trained models and the BreCaHAD NVIDIA pickle.
# Output is saved to project/gan-evaluation-results/gan_evaluation_metrics.txt

OUTPUT_FILE="project/gan-evaluation-results/gan_evaluation_metrics.txt"

echo "GAN Evaluation Metrics Calculation" > "$OUTPUT_FILE"
echo "==========================" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== 40x (snapshot 008800) ===" | tee -a "$OUTPUT_FILE"
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=project/gan-training-runs/40x/00005-stylegan2-breakhis_40x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-008800.pkl \
    --data=project/data/zips/breakhis_40x_train_256_crops.zip \
    2>&1 | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== 100x (snapshot 007800) ===" | tee -a "$OUTPUT_FILE"
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=project/gan-training-runs/100x/00000-stylegan2-breakhis_100x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-007800.pkl \
    --data=project/data/zips/breakhis_100x_train_256_crops.zip \
    2>&1 | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== 200x (snapshot 011600) ===" | tee -a "$OUTPUT_FILE"
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=project/gan-training-runs/200x/00001-stylegan2-breakhis_200x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-011600.pkl \
    --data=project/data/zips/breakhis_200x_train_256_crops.zip \
    2>&1 | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== 400x (snapshot 017000) ===" | tee -a "$OUTPUT_FILE"
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=project/gan-training-runs/400x/00000-stylegan2-breakhis_400x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-017000.pkl \
    --data=project/data/zips/breakhis_400x_train_256_crops.zip \
    2>&1 | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== BreCaHAD (NVIDIA pickle) ===" | tee -a "$OUTPUT_FILE"
python calc_metrics.py \
    --metrics=fid50k_full,kid50k_full,pr50k3_full \
    --network=project/data/brecahad/BreCaHAD-NVIDIA-pickle/brecahad.pkl \
    --data=project/data/brecahad/brecahad-crops-512.zip \
    2>&1 | tee -a "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Done. Results saved to $OUTPUT_FILE"
