#!/usr/bin/env python3
import argparse
import os
from run_mlp_inference import run_inference

def main():
    parser = argparse.ArgumentParser(
        description="Patch-based MLP spleen segmentation fusion"
    )
    parser.add_argument("--case_dir", required=True, type=str,
                        help="Path to a single ABNL-MARRO case folder")
    parser.add_argument("--model_path", default="/code/models/MLP_best.pkl",
                        help="Path to trained MLP model inside the container")
    args = parser.parse_args()

    if not os.path.exists(args.case_dir):
        raise FileNotFoundError(f"Case directory not found: {args.case_dir}")

    run_inference(args.case_dir, args.model_path)

if __name__ == "__main__":
    main()
