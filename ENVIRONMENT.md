# Environment record

Environment reported from the author's Termux installation
on 15 November 2026:

- Platform: Android / Termux ARM64
- Python: 3.14.10
- NumPy: 2.5.4
- Pillow: 12.3.0
- SciPy: 1.18.1

These versions were recorded after the benchmark run.
They are not an independently preserved snapshot of the run environment.

The benchmark imports argparse, csv, os and time from the
Python standard library.

OpenCV is not imported by benchmark_v28_no_opencv.py.

## Reproduction limitations

The exact 50-image manifest is not yet available.
The archive contains 10 image files whose correspondence
with benchmark image identifiers remains unverified.

Installing the listed versions alone does not establish
complete reproducibility of the reported results.

On Termux, package availability and installation methods
may differ from desktop Python environments.
