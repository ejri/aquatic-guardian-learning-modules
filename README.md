# Aquatic Guardian — Learn computer vision

Public notebooks and wildlife photographs for the Aquatic Guardian `#/learn` course.

Site: open `#/learn` on the Aquatic Guardian website. Each module’s **Open in Colab** button
points here, for example
[CV01 in Colab](https://colab.research.google.com/github/ejri/aquatic-guardian-learning-modules/blob/main/notebooks/CV01_images_are_arrays.ipynb).

## Notebooks

| File | Module |
|---|---|
| `notebooks/CV01_images_are_arrays.ipynb` | Images are arrays of numbers |
| `notebooks/CV02_how_water_changes_images.ipynb` | How water changes an image |
| `notebooks/CV03_classical_image_processing.ipynb` | Classical image processing |
| `notebooks/CV04_learning_to_classify.ipynb` | Learning to classify |
| `notebooks/CV05_robustness_and_reporting.ipynb` | Robustness and reporting |
| `notebooks/CV06_neural_networks_and_cnns.ipynb` | Neural networks and CNNs |
| `notebooks/CV07_detection_measurement_calibration.ipynb` | Detection, measurement, calibration |
| `notebooks/CV08_tracking_over_time.ipynb` | Tracking over time |
| `notebooks/CV09_from_pixels_to_claims.ipynb` | From pixels to claims |

In Colab: **File → Save a copy in Drive**, then **Runtime → Run all**. CPU is enough.
Every notebook can make its own synthetic underwater scene so graded answers have a known
truth. They also load **real** wildlife frames from `images/` (or download them from this
repository).

## Photographs

See `images/ATTRIBUTION.md`. Live-camera stills plus NOAA public-domain photographs
(green turtle, humpback whale) used for detection and classification examples.

## Source

`source/` regenerates the notebooks:

```bash
cd source
python cv01.py   # writes out/CV01_images_are_arrays.ipynb
```

## Licence

GPL-3.0. NOAA photographs in `images/noaa-*.jpg` are public domain (US government works).
