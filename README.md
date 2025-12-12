[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17905193.svg)](https://doi.org/10.5281/zenodo.17905193)

# Patch-Based MLP Spleen Segmentation Fusion

This repository provides a **GPU-accelerated patch-based multi-layer perceptron (MLP)** for voxel-wise fusion of multiple spleen segmentation pipelines.  
The model integrates **local CT intensity patches**, **local mask agreement/disagreement**, and **3D spatial coordinates** to produce an improved spleen segmentation.

A fully reproducible Apptainer/Singularity image is available on Zenodo.

---

## ⭐ Key Features

- Learned fusion of **four pipelines**:
  - TotalSegmentator  
  - DeepSpleenSeg  
  - DeepMultiOrgSeg  
  - GennUNet  
- CUDA-enabled voxel-wise patch inference  
- Dataset-agnostic: accepts **any folder structure** using `--mask_paths`  
- Includes a **working sample test case**  
- Reproducible: packaged as a .sif container (Zenodo DOI)

---

## 🧪 1. Quick Test (Recommended)

A complete working example is included:

test_case/
├── INPUTS/
│ └── image.nii.gz
└── OUTPUTS/
├── TotalSegmentator.nii.gz
├── DeepSpleenSeg.nii.gz
├── DeepMultiOrgSeg.nii.gz
└── GennUNet_spleen.nii.gz


### Run inference:
## 🚀 3. Run the MLP Fusion on Any Dataset (Singularity)

Use the following command exactly as shown.  
This binds your `/home-local` directory and runs inference on a chosen case:

```bash
ssingularity run --nv \
  -B /path/to/data:/path/to/data \
  mlp_fusion_v1.0.1.sif \
  --case_dir /path/to/data/case_folder \
  --mask_paths "OUTPUTS/TotalSegmentator.nii.gz,OUTPUTS/DeepSpleenSeg.nii.gz,OUTPUTS/DeepMultiOrgSeg.nii.gz,OUTPUTS/GennUNet_spleen.nii.gz"
```

After running, the fused MLP segmentation is written to:

/OUTPUTS/mlp_spleen_torch.nii.gz

🧩 4. Mask Path Options
Option A — Explicit list (recommended)
--mask_paths "OUTPUTS/A.nii.gz,OUTPUTS/B.nii.gz,OUTPUTS/C.nii.gz,OUTPUTS/D.nii.gz"

Option B — ABNL-MARRO default structure

If masks live under:

OUTPUTS/<PipelineName>/spleen.nii.gz


You may run:

--mask_paths ""


The script automatically loads all known pipeline directories.


🧠 5. Model Overview

Each voxel receives:

9×9×9 CT intensity patch

Four 9×9×9 mask patches

Normalized coordinates (x, y, z)

Total feature vector:```

729 (CT) + 4×729 (masks) + 3 (coords)

```
MLP architecture:

```[Input] → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid```



📦 6. Container & Zenodo
```
Latest version DOI (all versions):

https://doi.org/10.5281/zenodo.17903413

Direct .sif download (v1.0.1):
```
https://doi.org/10.5281/zenodo.17905193```