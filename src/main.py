#!/usr/bin/env python3
import argparse
import os
from run_mlp_inference import run_inference

def main():
    parser = argparse.ArgumentParser(
        description="Patch-based PyTorch MLP spleen segmentation fusion"
    )

    parser.add_argument(
        "--case_dir",
        required=True,
        type=str,
        help="Path to case folder (ABNL-MARRO or Task09_Spleen)"
    )

    parser.add_argument(
        "--model_path",
        default="/code/models/mlp_torch_best.pt",
        type=str,
        help="Path to trained PyTorch MLP model"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32768,
        help="Batch size used during inference"
    )

    parser.add_argument(
        "--mask_paths",
        type=str,
        default=None,
        help=(
            "Comma-separated list of mask paths relative to case_dir.\n"
            "Example:\n"
            "--mask_paths OUTPUTS/Total.nii.gz,OUTPUTS/Deep.nii.gz"
        )
    )

    args = parser.parse_args()

    # Parse CSV mask list → Python list
    if args.mask_paths is not None:
        mask_list = [x.strip() for x in args.mask_paths.split(",")]
    else:
        mask_list = None  # falls back to default ABNL-MARRO structure

    if not os.path.exists(args.case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {args.case_dir}")

    # Unified call
    run_inference(
        case_dir=args.case_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        custom_mask_paths=mask_list
    )


if __name__ == "__main__":
    main()
