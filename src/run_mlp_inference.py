#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 9
R = PATCH_SIZE // 2
PATCH_PIXELS = PATCH_SIZE ** 3

DEFAULT_PIPELINES = [
    "TotalSegmentator",
    "DeepSpleenSeg",
    "DeepMultiOrgSeg",
    "GennUNet",
]

# -------------------------------------------------------------
# Load masks when user provides --mask_paths
# -------------------------------------------------------------
def load_masks_custom(case_dir, mask_paths, shape):
    masks = []
    for rel_path in mask_paths:
        full = os.path.join(case_dir, rel_path)
        if os.path.exists(full):
            m = nib.load(full).get_fdata().astype(np.float32)
            m = (m == 1).astype(np.float32)
        else:
            print(f"⚠ Warning: Mask not found: {full} — using zeros.")
            m = np.zeros(shape, dtype=np.float32)
        masks.append(m)
    return masks

# -------------------------------------------------------------
# Load masks using ABNL-MARRO default folder structure
# -------------------------------------------------------------
def load_masks_default(case_dir, shape):
    masks = []
    for p in DEFAULT_PIPELINES:
        full = os.path.join(case_dir, "OUTPUTS", p, "spleen.nii.gz")
        if os.path.exists(full):
            m = nib.load(full).get_fdata().astype(np.float32)
            m = (m == 1).astype(np.float32)
        else:
            print(f"⚠ Warning: {p} mask not found — zeros used.")
            m = np.zeros(shape, dtype=np.float32)
        masks.append(m)
    return masks

# -------------------------------------------------------------
# MLP architecture
# -------------------------------------------------------------
class PatchMLP(torch.nn.Module):
    def __init__(self, in_features, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last, h))
            layers.append(torch.nn.ReLU())
            last = h
        layers.append(torch.nn.Linear(last, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------------------------------------------
# SAFE PATCH EXTRACTION (no boundary errors)
# -------------------------------------------------------------
def extract_patch_batch(volume, coords):
    patches = []
    valid_coords = []

    D, H, W = volume.shape

    for (x, y, z) in coords:
        # Check boundaries
        if (
            x - R < 0 or x + R >= D or
            y - R < 0 or y + R >= H or
            z - R < 0 or z + R >= W
        ):
            continue

        patch = volume[x-R:x+R+1, y-R:y+R+1, z-R:z+R+1]
        patches.append(patch.flatten())
        valid_coords.append((x, y, z))

    if len(patches) == 0:
        return None, []

    patches_tensor = torch.tensor(np.stack(patches), dtype=torch.float32, device=DEVICE)
    return patches_tensor, valid_coords


# -------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------------------
def run_inference(case_dir, model_path, batch_size=32768, custom_mask_paths=None):

    print(f"\n🔧 Using device: {DEVICE}")

    # ----------------------------------
    # Load CT
    # ----------------------------------
    ct_path = os.path.join(case_dir, "INPUTS", "image.nii.gz")
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata().astype(np.float32)
    D, H, W = ct.shape

    # ----------------------------------
    # Load masks (custom OR default)
    # ----------------------------------
    if custom_mask_paths is not None:
        print("🔹 Using user-specified mask paths")
        masks_np = load_masks_custom(case_dir, custom_mask_paths, ct.shape)
    else:
        print("🔹 Using default ABNL-MARRO pipeline masks")
        masks_np = load_masks_default(case_dir, ct.shape)

    # Region restriction
    union = np.clip(sum(masks_np), 0, 1).astype(bool)
    candidate_coords = np.argwhere(union)

    print(f"Candidate voxels: {len(candidate_coords)}")

    if len(candidate_coords) == 0:
        print("⚠ No candidates — output empty mask.")
        out = np.zeros(ct.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(out, ct_img.affine, ct_img.header),
                 os.path.join(case_dir, "OUTPUTS", "mlp_spleen_torch.nii.gz"))
        return

    # ----------------------------------
    # Load model
    # ----------------------------------
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = PatchMLP(ckpt["in_features"], ckpt["hidden_sizes"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_mask = np.zeros(ct.shape, dtype=np.uint8)
    coords_list = candidate_coords.tolist()

    # ----------------------------------
    # Batched inference
    # ----------------------------------
    for i in tqdm(range(0, len(coords_list), batch_size), desc="Predicting"):
        batch_coords = coords_list[i:i + batch_size]

        # Extract CT patches (safe, boundary-checked)
        ct_p, valid_coords = extract_patch_batch(ct, batch_coords)
        if ct_p is None:
            continue

        # Normalize CT patches
        ct_p = (ct_p - ct_p.mean(dim=1, keepdim=True)) / \
               (ct_p.std(dim=1, keepdim=True) + 1e-8)

        # Mask patches
        mask_feats = []
        for m in masks_np:
            mp, _ = extract_patch_batch(m, valid_coords)
            mask_feats.append(mp)

        # Coordinate features
        xyz = np.array(valid_coords)
        coords_norm = torch.tensor(
            np.stack([xyz[:, 0]/D, xyz[:, 1]/H, xyz[:, 2]/W], axis=1),
            dtype=torch.float32,
            device=DEVICE
        )

        # Combine all features
        X = torch.cat([ct_p] + mask_feats + [coords_norm], dim=1)

        with torch.no_grad():
            probs = torch.sigmoid(model(X)).cpu().numpy()

        # Write predictions
        for (x, y, z), p in zip(valid_coords, probs):
            out_mask[x, y, z] = 1 if p > 0.5 else 0

    # ----------------------------------
    # Save output mask
    # ----------------------------------
    save_path = os.path.join(case_dir, "OUTPUTS", "mlp_spleen_torch.nii.gz")
    nib.save(nib.Nifti1Image(out_mask.astype(np.uint8), ct_img.affine, ct_img.header), save_path)

    print(f"\n✓ Saved MLP prediction:\n{save_path}")
