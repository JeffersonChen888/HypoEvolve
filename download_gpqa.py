#!/usr/bin/env python3
"""Download GPQA Diamond dataset from Hugging Face."""

import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path("data/gpqa")
output_dir.mkdir(parents=True, exist_ok=True)

# Download the dataset using the command provided
print("Downloading GPQA Diamond dataset from Hugging Face...")
df = pd.read_parquet("hf://datasets/fingertap/GPQA-Diamond/test/gpqa_diamond.parquet")

# Save as parquet and CSV for convenience
parquet_path = output_dir / "gpqa_diamond.parquet"
csv_path = output_dir / "gpqa_diamond.csv"

df.to_parquet(parquet_path, index=False)
df.to_csv(csv_path, index=False)

print(f"Dataset downloaded successfully!")
print(f"  Parquet file: {parquet_path}")
print(f"  CSV file: {csv_path}")
print(f"\nDataset info:")
print(f"  Number of questions: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"\nFirst row sample:")
print(df.head(1).to_string())
