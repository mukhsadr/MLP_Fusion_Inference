#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import joblib

# ----------------------------
# Patch extraction
# ----------------------------
PATCH_SIZE = 9
R = PATCH_SIZE // 2

PIPELINES = [
    "TotalSegmentator",
    "DeepSpleenSeg",
    "DeepMultiOrgSeg",
    "GennUNet"
]

def extract_patch(volume, x, y, z):
    """
    Extract a (9×9×9) patch centered at (x,y,z). Returns flattened patch.
    """
    if (
        x - R < 0 or x + R >= volume.shape[0] or
        y - R < 0 or y + R >= volume.shape[1] or
        z - R < 0 or z + R >= volume.shape[2]
    ):
        return None
    
    patch = volume[
        x-R:x+R+1,
        y-R:y+R+1,
        z-R:z+R+1
    ]
    return patch.flatten()


def load_pipeline_masks(case_dir, shape):
    """
    Load each pipeline mask. If pipeline missing, returns zeros mask.
    """
    masks = []
    for p in PIPELINES:
        mask_path = os.path.join(case_dir, "OUTPUTS", p, "spleen.nii.gz")
        if os.path.exists(mask_path):
            m = nib.load(mask_path).get_fdata()
            m = (m == 1).astype(np.float32)
        else:
            m = np.zeros(shape, dtype=np.float32)
        masks.append(m)
    return masks


def run_inference(case_dir, model_path):
    """
    Performs voxel-wise inference.
    """
    input_img_path = os.path.join(case_dir, "INPUTS", "image.nii.gz")
    ct_img = nib.load(input_img_path)
    ct = ct_img.get_fdata()

    # Load masks
    masks = load_pipeline_masks(case_dir, ct.shape)

    # Load trained MLP
    clf = joblib.load(model_path)

    # Output array
    out_mask = np.zeros(ct.shape, dtype=np.uint8)

    # Iterate through all voxels
    print("Running inference…")
    for x in tqdm(range(R, ct.shape[0]-R), desc="X"):
        for y in range(R, ct.shape[1]-R):
            for z in range(R, ct.shape[2]-R):
                
                # --- CT patch ---
                f = []
                patch_ct = extract_patch(ct, x, y, z)
                if patch_ct is None:
                    continue
                f.append(patch_ct)

                # --- Mask patches ---
                valid = True
                for m in masks:
                    patch_m = extract_patch(m, x, y, z)
                    if patch_m is None:
                        valid = False
                        break
                    f.append(patch_m)
                if not valid:
                    continue

                # --- Stack ---
                feature = np.concatenate(f).reshape(1, -1)

                # --- Predict ---
                prob = clf.predict_proba(feature)[0, 1]
                out_mask[x, y, z] = 1 if prob > 0.5 else 0

    # Save output mask
    out_path = os.path.join(case_dir, "OUTPUTS", "mlp_spleen.nii.gz")
    out_img = nib.Nifti1Image(out_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(out_img, out_path)

    print(f"✓ Saved MLP segmentation: {out_path}")
