"""Authenticity check for a trained StyleGAN2 model.

Implements the Authenticity metric from Alaa et al. (2022),
"How Faithful is your Synthetic Data? Sample-level Metrics for
Evaluating and Auditing Generative Models" (ICML 2022).

For each generated image, checks whether it is no farther from its
nearest training image than that training image is from its own nearest
other training image. If so, the generated image is flagged as
"unauthentic" (a potential local copy). The Authenticity score is the
fraction of generated images that are authentic (not flagged). Higher
is better.

Two variants are computed:
  1. Standard: real-to-real baseline excludes only self (as in the paper)
  2. Strict: real-to-real baseline excludes all crops from the same
     source image (BreaKHis-specific adaptation, since overlapping crops
     from the same source share pixels and are artificially close)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import legacy


def load_generator(network_pkl, device):
    """Load the G_ema generator from a .pkl snapshot."""
    print(f'Loading generator from {network_pkl}...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G


def load_vgg16(device):
    """Load the same VGG16 used by the precision/recall metrics."""
    url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl'
    print('Loading VGG16 feature extractor...')
    with dnnlib.util.open_url(url) as f:
        vgg16 = pickle.load(f).to(device)
    return vgg16


def generate_images(G, num_images, device, seed=0):
    """Generate images from the model, split evenly across classes if conditional."""
    print(f'Generating {num_images} images...')
    images = []
    labels = []
    rng = np.random.RandomState(seed)

    if G.c_dim > 0:
        num_classes = G.c_dim
        per_class = num_images // num_classes
        for class_idx in range(num_classes):
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, class_idx] = 1
            for i in range(per_class):
                z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device).float()
                img = G(z, label, truncation_psi=1, noise_mode='const')
                img_uint8 = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images.append(img_uint8[0].cpu().numpy())
                labels.append(class_idx)
        for i in range(num_images - per_class * num_classes):
            class_idx = i % num_classes
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, class_idx] = 1
            z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device).float()
            img = G(z, label, truncation_psi=1, noise_mode='const')
            img_uint8 = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img_uint8[0].cpu().numpy())
            labels.append(class_idx)
    else:
        for i in range(num_images):
            z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device).float()
            label = torch.zeros([1, G.c_dim], device=device)
            img = G(z, label, truncation_psi=1, noise_mode='const')
            img_uint8 = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img_uint8[0].cpu().numpy())
            labels.append(0)

    return images, labels


def load_real_images(dataset_zip, max_images=None):
    """Load real images from the training dataset ZIP."""
    print(f'Loading real images from {dataset_zip}...')
    import zipfile
    import io

    images = []
    filenames = []
    with zipfile.ZipFile(dataset_zip, 'r') as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if max_images is not None:
            names = names[:max_images]
        for name in names:
            with zf.open(name) as f:
                img = PIL.Image.open(io.BytesIO(f.read())).convert('RGB')
                images.append(np.array(img))
                filenames.append(name)

    print(f'  Loaded {len(images)} real images.')
    return images, filenames


def extract_features(vgg16, images, device, batch_size=32):
    """Extract VGG16 features for a list of numpy images (H, W, 3) uint8."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            feat = vgg16(batch_tensor, return_features=True)
        features.append(feat.cpu())
        if (i // batch_size) % 10 == 0:
            print(f'  Extracting features: {min(i + batch_size, len(images))}/{len(images)}')
    return torch.cat(features, dim=0)


def find_nearest_neighbours(query_features, reference_features, batch_size=100):
    """For each query, find the nearest reference by L2 distance."""
    nn_distances = []
    nn_indices = []
    for i in range(0, query_features.shape[0], batch_size):
        batch = query_features[i:i+batch_size]
        dists = torch.cdist(batch, reference_features)
        min_dists, min_idxs = dists.min(dim=1)
        nn_distances.append(min_dists)
        nn_indices.append(min_idxs)
    nn_distances = torch.cat(nn_distances)
    nn_indices = torch.cat(nn_indices)
    return nn_distances.numpy(), nn_indices.numpy()


def compute_real_to_real_nn(real_features, batch_size=100, crops_per_image=None):
    """For each real image, find its nearest OTHER real image.

    Returns:
      - nn_any: nearest neighbour distance excluding only self
      - nn_cross: nearest neighbour distance excluding all crops from
                  same source (None if crops_per_image not set)
    """
    num_real = real_features.shape[0]

    print('Computing real→real NN distances (excluding self)...')
    nn_any = []
    for i in range(0, num_real, batch_size):
        batch = real_features[i:i+batch_size]
        dists = torch.cdist(batch, real_features)
        for j in range(dists.shape[0]):
            dists[j, i + j] = float('inf')
        min_dists, _ = dists.min(dim=1)
        nn_any.append(min_dists)
    nn_any = torch.cat(nn_any).numpy()

    nn_cross = None
    if crops_per_image is not None and crops_per_image > 1:
        print(f'Computing real→real NN distances (excluding same source, {crops_per_image} crops/image)...')
        nn_cross = []
        for i in range(0, num_real, batch_size):
            batch = real_features[i:i+batch_size]
            actual_bs = batch.shape[0]
            dists = torch.cdist(batch, real_features)
            for j in range(actual_bs):
                global_idx = i + j
                source_id = global_idx // crops_per_image
                source_start = source_id * crops_per_image
                source_end = min(source_start + crops_per_image, num_real)
                dists[j, source_start:source_end] = float('inf')
            min_dists, _ = dists.min(dim=1)
            nn_cross.append(min_dists)
        nn_cross = torch.cat(nn_cross).numpy()

    return nn_any, nn_cross


def compute_authenticity(gen_to_real_distances, gen_to_real_indices,
                         real_to_real_distances):
    """Compute Authenticity score (Alaa et al., 2022, Section 3.2.2).

    For each generated sample j:
      - d_j = gen_to_real_distances[j] = distance from j to its nearest
        real training image x*
      - x* index = gen_to_real_indices[j]
      - d_{x*} = real_to_real_distances[x*] = distance from x* to its
        nearest other real training image

    A generated sample j is "authentic" if it is strictly farther from
    x* than x* is to its own nearest real neighbour:

      authentic[j] = d_j > d_{x*}

    Equivalently, j is "unauthentic" (a possible copy) if d_j <= d_{x*},
    i.e., j sits inside x*'s local real neighbourhood. This matches the
    statistic a_j = 1{d_{x,i*} <= d_{r,i*}} in Alaa et al. (2022), p.6.

    Authenticity score = fraction of generated samples that are authentic.
    Higher is better (1.0 = no copying detected).

    Returns (authenticity_score, num_unauthentic).
    """
    real_baseline = real_to_real_distances[gen_to_real_indices]
    authentic = gen_to_real_distances > real_baseline
    num_unauthentic = int(np.sum(~authentic))
    authenticity = float(np.mean(authentic))
    return authenticity, num_unauthentic


def main():
    parser = argparse.ArgumentParser(
        description='Authenticity check (Alaa et al., 2022)')
    parser.add_argument('--network', required=True,
                        help='Path to trained model .pkl')
    parser.add_argument('--data', required=True,
                        help='Path to training dataset .zip')
    parser.add_argument('--outdir', required=True,
                        help='Output directory')
    parser.add_argument('--num-gen', type=int, default=1000,
                        help='Number of images to generate (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction (default: 32)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--crops-per-image', type=int, default=None,
                        help='Number of crops per source image (e.g. 6 for BreaKHis). '
                             'When set, also computes a strict variant that excludes '
                             'all crops from the same source image.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

    # Load model and VGG16
    G = load_generator(args.network, device)
    vgg16 = load_vgg16(device)

    # Generate images
    gen_images, gen_labels = generate_images(G, args.num_gen, device, seed=args.seed)

    # Load real training images
    real_images, real_filenames = load_real_images(args.data)

    # Extract features
    print('Extracting features for generated images...')
    gen_features = extract_features(vgg16, gen_images, device, batch_size=args.batch_size)
    print('Extracting features for real images...')
    real_features = extract_features(vgg16, real_images, device, batch_size=args.batch_size)

    # Find nearest real training image for each generated image
    print('Computing nearest neighbours (generated → real)...')
    gen_to_real_distances, gen_to_real_indices = find_nearest_neighbours(
        gen_features, real_features)

    # Compute real-to-real NN distances (the local neighbourhood radii)
    real_nn_any, real_nn_cross = compute_real_to_real_nn(
        real_features, crops_per_image=args.crops_per_image)

    # Compute Authenticity — standard (as in Alaa et al.)
    auth_standard, num_unauth_standard = compute_authenticity(
        gen_to_real_distances, gen_to_real_indices, real_nn_any)

    # Compute Authenticity — strict (excluding same-source crops)
    auth_strict, num_unauth_strict = None, None
    if real_nn_cross is not None:
        auth_strict, num_unauth_strict = compute_authenticity(
            gen_to_real_distances, gen_to_real_indices, real_nn_cross)

    # Print and save results
    num_real = len(real_nn_any)
    num_gen = len(gen_to_real_distances)
    lines = []
    lines.append('Authenticity Check Results (Alaa et al., 2022)')
    lines.append('=' * 60)
    lines.append(f'Generated images: {num_gen}')
    lines.append(f'Training images:  {num_real}')
    lines.append('')
    lines.append('Standard (excluding self only, as in Alaa et al.):')
    lines.append(f'  Authenticity:  {auth_standard:.4f}')
    lines.append(f'  Unauthentic:   {num_unauth_standard} / {num_gen}')
    if auth_strict is not None:
        lines.append('')
        lines.append('Strict (excluding same-source crops, BreaKHis adaptation):')
        lines.append(f'  Authenticity:  {auth_strict:.4f}')
        lines.append(f'  Unauthentic:   {num_unauth_strict} / {num_gen}')
    lines.append('')
    lines.append('Generated→Real NN distances:')
    lines.append(f'  min:    {np.min(gen_to_real_distances):.4f}')
    lines.append(f'  median: {np.median(gen_to_real_distances):.4f}')
    lines.append(f'  mean:   {np.mean(gen_to_real_distances):.4f}')
    lines.append('')
    lines.append('Real→Real NN distances (standard, excluding self):')
    lines.append(f'  min:    {np.min(real_nn_any):.4f}')
    lines.append(f'  median: {np.median(real_nn_any):.4f}')
    lines.append(f'  mean:   {np.mean(real_nn_any):.4f}')
    if real_nn_cross is not None:
        lines.append('')
        lines.append('Real→Real NN distances (strict, excluding same source):')
        lines.append(f'  min:    {np.min(real_nn_cross):.4f}')
        lines.append(f'  median: {np.median(real_nn_cross):.4f}')
        lines.append(f'  mean:   {np.mean(real_nn_cross):.4f}')

    output = '\n'.join(lines)
    print(f'\n{output}')

    path = os.path.join(args.outdir, 'authenticity_results.txt')
    with open(path, 'w') as f:
        f.write(output + '\n')
    print(f'\nSaved results to {path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
