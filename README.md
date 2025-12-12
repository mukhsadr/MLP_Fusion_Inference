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
### Structure of expected directory for test run:
## 🧪 1. Quick Test (Recommended)

A complete working example is included:

```
test_case/
├── INPUTS/
│ └── image.nii.gz
└── OUTPUTS/
  └──TotalSegmentator.nii.gz
  └──DeepSpleenSeg.nii.gz
  └──DeepMultiOrgSeg.nii.gz
  └──GennUNet_spleen.nii.gz
```


### Run inference:
## 🚀 3. Run the MLP Fusion on Any Dataset (Singularity)

Use the following command exactly as shown.  
This binds your `/home-local` directory and runs inference on a chosen case:

```
singularity run --nv   -B /home-local:/home-local   mlp_fusion_v1.0.1.sif   --case_dir  test_case   --mask_paths "OUTPUTS/TotalSegmentator.nii.gz,OUTPUTS/DeepSpleenSeg.nii.gz,OUTPUTS/DeepMultiOrgSeg.nii.gz,OUTPUTS/GennUNet_spleen.nii.gz
```


```bash
singularity run --nv \
  -B /path/to/data:/path/to/data \
  mlp_fusion_v1.0.1.sif \
  --case_dir /path/to/data/case_folder \
  --mask_paths "OUTPUTS/TotalSegmentator.nii.gz,OUTPUTS/DeepSpleenSeg.nii.gz,OUTPUTS/DeepMultiOrgSeg.nii.gz,OUTPUTS/GennUNet_spleen.nii.gz"
```

After running, the fused MLP segmentation is written to:

/OUTPUTS/mlp_spleen_torch.nii.gz



### Download singularity image sif:
## 📦 4. Container & Zenodo


Latest version DOI (all versions):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17905193.svg)](https://doi.org/10.5281/zenodo.17905193)


```
https://doi.org/10.5281/zenodo.17903413
```
