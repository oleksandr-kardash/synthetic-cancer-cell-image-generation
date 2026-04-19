#!/bin/bash
# Run the Alaa et al. (2022) Authenticity check for the best snapshot of each magnification.
# Results are saved to project/gan-evaluation-results/authenticity-check/{magnification}-snapshot-{id}/

echo "=== 40x (snapshot 008800) ==="
python authenticity_check.py \
    --network project/gan-training-runs/40x/00005-stylegan2-breakhis_40x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-008800.pkl \
    --data project/data/zips/breakhis_40x_train_256_crops.zip \
    --outdir project/gan-evaluation-results/authenticity-check/40x-snapshot-008800 \
    --crops-per-image 6
echo ""

echo "=== 100x (snapshot 007800) ==="
python authenticity_check.py \
    --network project/gan-training-runs/100x/00000-stylegan2-breakhis_100x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-007800.pkl \
    --data project/data/zips/breakhis_100x_train_256_crops.zip \
    --outdir project/gan-evaluation-results/authenticity-check/100x-snapshot-007800 \
    --crops-per-image 6
echo ""

echo "=== 200x (snapshot 011600) ==="
python authenticity_check.py \
    --network project/gan-training-runs/200x/00001-stylegan2-breakhis_200x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-011600.pkl \
    --data project/data/zips/breakhis_200x_train_256_crops.zip \
    --outdir project/gan-evaluation-results/authenticity-check/200x-snapshot-011600 \
    --crops-per-image 6
echo ""

echo "=== 400x (snapshot 017000) ==="
python authenticity_check.py \
    --network project/gan-training-runs/400x/00000-stylegan2-breakhis_400x_train_256_crops-gpus1-batch16-gamma0.8192/network-snapshot-017000.pkl \
    --data project/data/zips/breakhis_400x_train_256_crops.zip \
    --outdir project/gan-evaluation-results/authenticity-check/400x-snapshot-017000 \
    --crops-per-image 6
echo ""

echo "Done. Results saved to project/gan-evaluation-results/authenticity-check/"
