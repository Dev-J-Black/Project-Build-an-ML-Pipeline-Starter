#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import os
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


# DO NOT MODIFY
def go(args):
    # Single W&B run; log params for reproducibility
    run = wandb.init(job_type="basic_cleaning", save_code=True)
    run.config.update(vars(args))

    # Download input artifact (expects a single CSV file)
    logger.info(f"Fetching input artifact: {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    local_path = artifact.file()  # works when the artifact is a single file

    if not local_path or not os.path.exists(local_path):
        raise FileNotFoundError("Could not locate the CSV from the input artifact.")

    logger.info(f"Reading CSV: {local_path}")
    df = pd.read_csv(local_path)

    # ---- Basic cleaning aligned with rubric ----
    # 1) Price range filter
    logger.info(f"Filtering price between {args.min_price} and {args.max_price}")
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    # 2) NYC geo-bounds
    if {"longitude", "latitude"}.issubset(df.columns):
        logger.info("Applying NYC latitude/longitude bounds")
        lon_ok = df["longitude"].between(-74.25, -73.50)
        lat_ok = df["latitude"].between(40.5, 41.2)
        df = df[lon_ok & lat_ok].copy()

    # 3) Parse last_review robustly
    if "last_review" in df.columns:
        logger.info("Parsing last_review to datetime")
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # 4) (optional but safe) drop obvious NAs used downstream
    key_cols = [c for c in ["price", "latitude", "longitude"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols)

    # Save and log cleaned artifact
    out_csv = "clean_sample.csv"
    logger.info(f"Writing cleaned CSV: {out_csv} (shape={df.shape})")
    df.to_csv(out_csv, index=False)

    logger.info("Uploading cleaned artifact to W&B")
    artifact_out = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact_out.add_file(out_csv)
    run.log_artifact(artifact_out)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    # All artifact identifiers are strings; price thresholds are floats
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified W&B artifact to read, e.g. 'user/proj/sample.csv:latest'",
        required=True,

    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned CSV artifact to create, e.g. 'clean_sample.csv'",
        required=True,

    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Artifact type for the cleaned dataset, e.g. 'clean_sample'",
        required=True,

    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Short description of what the cleaned artifact contains",
        required=True,

    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum nightly price to keep (rows below are dropped)",
        required=True,

    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum nightly price to keep (rows above are dropped)",
        required=True,

    )

    args = parser.parse_args()
    go(args)
