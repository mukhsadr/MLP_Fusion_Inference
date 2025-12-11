# Patch-Based MLP Spleen Segmentation Fusion

This repository contains the inference container for our **patch-based, voxel-wise multi-layer perceptron (MLP)** that fuses multiple publicly available spleen/organ segmentation pipelines.

This model performs **learned fusion**, taking CT intensities and four pipeline masks as input, producing an improved spleen segmentation.

---

## 🔧 1. Requirements Before Running the MLP

For each case, you must first generate spleen segmentations using:

- TotalSegmentator  
- DeepSpleenSeg  
- DeepMultiOrgSeg  
- GennUNet  

Place results in:

