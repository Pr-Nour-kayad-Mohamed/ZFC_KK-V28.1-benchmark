# V28.1 — Exploratory benchmark on 50 images

Initial GitHub prerelease of the ZFC_KK V28.1 exploratory benchmark.

## Evaluation
- 50 real source images with added synthetic noise.
- 5 random seeds and 4 noise levels.
- 3 methods: V28.1, Bilateral_approx and NLM_simplified.
- 3,000 result rows.
- PSNR, SSIM and processing-time measurements.

## Observations
V28.1 has the highest mean PSNR_Y and SSIM_Y among the
three tested implementations at noise levels 0.15 and 0.25.
At 0.05, Bilateral_approx leads on both metrics.
At 0.10, Bilateral_approx leads on PSNR_Y and V28.1 on SSIM_Y.

These results do not establish general superiority.

## Included materials
Benchmark code, CSV results, environment documentation,
citation metadata, original benchmark ZIP and checksums.

## Limitations
The results cover 50 images, but the original ZIP includes
only 10 image files. Their correspondence with benchmark
image identifiers remains unverified.

Complete reproduction of the reported run is not yet established.
Image provenance and redistribution permissions remain to be documented.
No new license is granted by these release notes.

## Associated Zenodo record
https://doi.org/10.5281/zenodo.24238218

The GitHub release includes additional documentation.
It is not asserted to be identical to the Zenodo archive.
