# ZFC_KK V28.1 — Exploratory Image Benchmark

Author: Mohamed NOUR KAYAD

## Overview

Exploratory evaluation of the ZFC_KK V28.1 prototype under Termux ARM64.

The reported benchmark covers 50 real source images with synthetic noise:
- 5 random seeds;
- 4 noise levels;
- 3 evaluated methods;
- 3,000 result rows.

Compared methods:
- V28.1;
- Bilateral_approx;
- NLM_simplified.

Metrics recorded:
- PSNR_Y and SSIM_Y;
- PSNR_RGB and SSIM_RGB;
- processing time.

## Main observations

According to the exported summary:
- At noise level 0.05, Bilateral_approx has the highest mean PSNR_Y and SSIM_Y.
- At noise level 0.10, Bilateral_approx has the highest mean PSNR_Y, while V28.1 has the highest mean SSIM_Y.
- At noise levels 0.15 and 0.25, V28.1 has the highest mean PSNR_Y and SSIM_Y among the three tested implementations.

These observations apply only to this benchmark configuration.
They do not establish general superiority over image-denoising methods.

## Files

- benchmark_v28_no_opencv.py: benchmark implementation.
- resultats_no_opencv/: detailed and summary CSV results.
- horodatage_v28_1.zip: archived benchmark package, including 10 image files.
- SHA-256 text files: recorded checksums.

## Dataset limitations

The results cover 50 source images, but the archive includes only 10 image files.

Those 10 files were copied from the Pictures directory using filename sorting.
Their correspondence with benchmark image identifiers has not yet been verified.

The full 50-image dataset and a verified image-ID-to-filename manifest
are not currently included. Complete reproduction of the reported run
is therefore not yet established.

## Methodological limitations

The reference methods are local approximate or simplified implementations.
The metric implementations and preprocessing should be inspected in the code.
The results concern synthetic noise added to real images, not measurements
on naturally noisy images or astronomical observations.

## Execution used for the reported run

python -u benchmark_v28_no_opencv.py --images /storage/emulated/0/Pictures --max-images 50 --n-seeds 5

This command uses a device-specific image directory.
Using a different image collection will not reproduce the same results.

## Zenodo record

https://doi.org/10.5281/zenodo.24238218

## Image rights

Inclusion of an image does not establish ownership or permission for reuse.
Image provenance and redistribution permissions remain to be documented.
