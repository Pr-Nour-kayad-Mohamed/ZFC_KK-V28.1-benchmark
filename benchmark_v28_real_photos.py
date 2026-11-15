#!/usr/bin/env python3
# benchmark_v28_real_photos.py
# Banc V28.1 : YCbCr, 20 graines, PSNR/SSIM, bilatéral et NLM.
# Usage :
#   python benchmark_v28_real_photos.py
#   python benchmark_v28_real_photos.py --images ./photos --out ./resultats

import argparse
import csv
import os
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


# -----------------------------
# Métriques
# -----------------------------

def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b, data_range=255.0):
    error = mse(a, b)
    if error == 0:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / error))


def ssim_gray(a, b, data_range=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    if HAS_SKIMAGE:
        return float(skimage_ssim(
            a, b,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        ))

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)

    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a2 = gaussian_filter(a ** 2, sigma=1.5) - mu_a2
    sigma_b2 = gaussian_filter(b ** 2, sigma=1.5) - mu_b2
    sigma_ab = gaussian_filter(a * b, sigma=1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )

    return float(np.mean(ssim_map))


# -----------------------------
# Estimateur de bruit
# -----------------------------

def estimate_noise_mad_haar(img):
    img = img.astype(np.float64)
    h, w = img.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    sub = img[:h2, :w2]

    hh = 0.5 * (
        sub[0::2, 0::2]
        - sub[0::2, 1::2]
        - sub[1::2, 0::2]
        + sub[1::2, 1::2]
    )

    mad = np.median(np.abs(hh - np.median(hh)))
    return float(mad / 0.6744897501960817)


# -----------------------------
# V28.1 : luminance uniquement
# -----------------------------

def v28_local_2d(img):
    img = img.astype(np.float64)
    sigma_est = estimate_noise_mad_haar(img)

    if sigma_est < 1e-4:
        return img.copy(), {
            "sigma_est": sigma_est,
            "mean_theta": 0.0,
            "flat_ratio": 0.0
        }

    smooth_guide = gaussian_filter(
        img, sigma=max(0.8, 1.3 * sigma_est)
    )

    gx = sobel(smooth_guide, axis=1) / 4.0
    gy = sobel(smooth_guide, axis=0) / 4.0
    grad_mag = np.hypot(gx, gy)

    noise_grad_floor = 0.35 * sigma_est
    excess_grad = np.maximum(grad_mag - noise_grad_floor, 0.0)
    k_edge = max(1.6 * sigma_est, 0.03)

    w_flat = np.exp(-((excess_grad / k_edge) ** 2))

    s_coarse = gaussian_filter(img, sigma=1.10)
    s_fine = gaussian_filter(img, sigma=0.35)

    s_adaptive = w_flat * s_coarse + (1.0 - w_flat) * s_fine

    theta_max = min(
        0.30 * ((sigma_est / 0.25) ** 2),
        0.35
    )

    theta_map = theta_max * w_flat

    mu_local = gaussian_filter(s_adaptive, sigma=2.5)
    local_std = np.sqrt(
        gaussian_filter((s_adaptive - mu_local) ** 2, sigma=2.5)
    )

    local_scale = np.maximum(
        local_std * 2.5,
        1.5 * sigma_est
    )

    q_local = (s_adaptive - mu_local) / np.maximum(local_scale, 1e-6)

    out = (
        np.cos(theta_map) * s_adaptive
        - np.sin(theta_map) * q_local
    )

    return out, {
        "sigma_est": sigma_est,
        "mean_theta": float(np.mean(theta_map)),
        "flat_ratio": float(np.mean(w_flat))
    }


# -----------------------------
# Méthodes concurrentes
# -----------------------------

def bilateral_filter(img):
    if not HAS_CV2:
        raise RuntimeError("OpenCV est requis pour le filtre bilatéral.")

    img8 = np.clip(img, 0, 255).astype(np.uint8)
    out = cv2.bilateralFilter(
        img8,
        d=9,
        sigmaColor=75,
        sigmaSpace=75,
        borderType=cv2.BORDER_REFLECT
    )
    return out.astype(np.float64)


def non_local_means(img):
    if not HAS_CV2:
        raise RuntimeError("OpenCV est requis pour Non-Local Means.")

    img8 = np.clip(img, 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoising(
        img8,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
        normType=cv2.NORM_L2
    )
    return out.astype(np.float64)


# -----------------------------
# Conversion couleur
# -----------------------------

def rgb_to_ycbcr(rgb):
    rgb = rgb.astype(np.float64)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return np.stack([y, cb, cr], axis=2)


def ycbcr_to_rgb(ycbcr):
    y = ycbcr[:, :, 0]
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)

    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb, 0, 255)


# -----------------------------
# Jeu de contrôle synthétique
# -----------------------------

def make_control_images(n_images=5, size=256):
    images = []
    yy, xx = np.mgrid[0:size, 0:size] / float(size)

    for k in range(n_images):
        base = 0.35 + 0.30 * np.sin(6.0 * np.pi * xx + k)
        base += 0.20 * np.cos(5.0 * np.pi * yy + 0.7 * k)
        base += 0.10 * np.sin(
            18.0 * np.pi * np.hypot(xx - 0.5, yy - 0.5)
        )

        base = 0.5 + 0.25 * np.tanh(base)

        rgb = np.stack(
            [base, 0.85 * base + 0.05, 0.75 * base + 0.10],
            axis=2
        )

        rgb = np.clip(255.0 * rgb, 0, 255)
        images.append(rgb.astype(np.uint8))

    return images


# -----------------------------
# Banc principal
# -----------------------------

def load_images(image_dir, max_images=20, size=256):
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []

    if image_dir and os.path.isdir(image_dir):
        for name in sorted(os.listdir(image_dir)):
            if name.lower().endswith(extensions):
                files.append(os.path.join(image_dir, name))

    images = []

    for path in files[:max_images]:
        try:
            img = Image.open(path).convert("RGB")

            if img.size != (size, size):
                img = img.resize((size, size), Image.LANCZOS)

            images.append(np.asarray(img, dtype=np.uint8))
        except Exception as error:
            print(f"Image ignorée : {path} ({error})")

    return images


def run_benchmark(args):
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "v28_1_results.csv")
    summary_path = os.path.join(args.out, "v28_1_summary.csv")

    images = load_images(
        args.images,
        max_images=args.max_images,
        size=args.size
    )

    if not images:
        print("Aucune image trouvée. Génération d’un jeu de contrôle synthétique.")
        images = make_control_images(
            n_images=args.max_images,
            size=args.size
        )

    noise_levels = [0.05, 0.10, 0.15, 0.25]
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    methods = ["V28.1", "Bilateral", "NLM"]

    fieldnames = [
        "image_id",
        "seed",
        "noise_sigma",
        "method",
        "PSNR_Y",
        "SSIM_Y",
        "PSNR_RGB",
        "SSIM_RGB",
        "time_ms",
        "sigma_est",
        "mean_theta",
        "flat_ratio"
    ]

    rows = []

    for image_id, clean_rgb in enumerate(images):
        clean_ycbcr = rgb_to_ycbcr(clean_rgb)
        clean_y = clean_ycbcr[:, :, 0]

        for sigma in noise_levels:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                noise = rng.normal(
                    0.0,
                    sigma * 255.0,
                    size=clean_y.shape
                )

                noisy_y = np.clip(clean_y + noise, 0, 255)
                noisy_ycbcr = clean_ycbcr.copy()
                noisy_ycbcr[:, :, 0] = noisy_y

                for method in methods:
                    start = time.perf_counter()

                    if method == "V28.1":
                        denoised_y, metadata = v28_local_2d(noisy_y)
                    elif method == "Bilateral":
                        denoised_y = bilateral_filter(noisy_y)
                        metadata = {
                            "sigma_est": "",
                            "mean_theta": "",
                            "flat_ratio": ""
                        }
                    else:
                        denoised_y = non_local_means(noisy_y)
                        metadata = {
                            "sigma_est": "",
                            "mean_theta": "",
                            "flat_ratio": ""
                        }

                    elapsed_ms = (
                        time.perf_counter() - start
                    ) * 1000.0

                    denoised_ycbcr = noisy_ycbcr.copy()
                    denoised_ycbcr[:, :, 0] = denoised_y
                    denoised_rgb = ycbcr_to_rgb(denoised_ycbcr)

                    row = {
                        "image_id": image_id,
                        "seed": seed,
                        "noise_sigma": sigma,
                        "method": method,
                        "PSNR_Y": psnr(clean_y, denoised_y, 255.0),
                        "SSIM_Y": ssim_gray(clean_y, denoised_y, 255.0),
                        "PSNR_RGB": psnr(clean_rgb, denoised_rgb, 255.0),
                        "SSIM_RGB": ssim_gray(
                            clean_rgb.mean(axis=2),
                            denoised_rgb.mean(axis=2),
                            255.0
                        ),
                        "time_ms": elapsed_ms,
                        "sigma_est": metadata.get("sigma_est", ""),
                        "mean_theta": metadata.get("mean_theta", ""),
                        "flat_ratio": metadata.get("flat_ratio", "")
                    }

                    rows.append(row)

        print(f"Image {image_id + 1}/{len(images)} terminée.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for sigma in noise_levels:
        for method in methods:
            subset = [
                r for r in rows
                if r["noise_sigma"] == sigma
                and r["method"] == method
            ]

            if not subset:
                continue

            psnr_y = np.array([r["PSNR_Y"] for r in subset])
            ssim_y = np.array([r["SSIM_Y"] for r in subset])
            psnr_rgb = np.array([r["PSNR_RGB"] for r in subset])
            ssim_rgb = np.array([r["SSIM_RGB"] for r in subset])
            times = np.array([r["time_ms"] for r in subset])

            summary_rows.append({
                "noise_sigma": sigma,
                "method": method,
                "n_runs": len(subset),
                "PSNR_Y_mean": np.mean(psnr_y),
                "PSNR_Y_std": np.std(psnr_y, ddof=1),
                "PSNR_Y_q05": np.quantile(psnr_y, 0.05),
                "PSNR_Y_q95": np.quantile(psnr_y, 0.95),
                "SSIM_Y_mean": np.mean(ssim_y),
                "SSIM_Y_std": np.std(ssim_y, ddof=1),
                "SSIM_Y_q05": np.quantile(ssim_y, 0.05),
                "SSIM_Y_q95": np.quantile(ssim_y, 0.95),
                "PSNR_RGB_mean": np.mean(psnr_rgb),
                "SSIM_RGB_mean": np.mean(ssim_rgb),
                "time_ms_mean": np.mean(times),
                "time_ms_median": np.median(times),
                "time_ms_q95": np.quantile(times, 0.95)
            })

    summary_fields = list(summary_rows[0].keys())

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("
Banc terminé.")
    print(f"Résultats détaillés : {csv_path}")
    print(f"Résumé statistique : {summary_path}")

    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Banc V28.1 sur images réelles en YCbCr."
    )

    parser.add_argument(
        "--images",
        default="./photos",
        help="Dossier contenant les images RGB."
    )

    parser.add_argument(
        "--out",
        default="./resultats",
        help="Dossier de sortie."
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Taille carrée des images."
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Nombre maximal d’images."
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Nombre de réalisations de bruit par configuration."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Graine initiale."
    )

    args = parser.parse_args()

    if not HAS_CV2:
        pr
