import argparse
import pandas as pd
from pathlib import Path

# HAM10000 7 classes
TARGET_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}

# ISIC 2019 mapping to HAM10000
ISIC_MAPPING = {
    "MEL": "mel",
    "NV": "nv",
    "BCC": "bcc",
    "AK": "akiec",
    "BKL": "bkl",
    "DF": "df",
    "VASC": "vasc",
    "SCC": "scc", # Will be filtered out
    "UNK": "unk"  # Unknown, will be filtered
}

# PAD-UFES-20 mapping to HAM10000
PAD_MAPPING = {
    "ACK": "akiec",
    "BCC": "bcc",
    "MEL": "mel",
    "NEV": "nv",
    "SCC": "scc", # Will be filtered
    "SEK": "bkl", # Seborrheic Keratosis maps to BKL
}

def process_isic(isic_csv_path: Path) -> pd.DataFrame:
    if not isic_csv_path.exists():
        print(f"Warning: ISIC CSV not found at {isic_csv_path}. Skipping.")
        return pd.DataFrame()
    
    df = pd.read_csv(isic_csv_path)
    # ISIC 2019 ground truth is one-hot encoded
    # Columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    
    processed = []
    for _, row in df.iterrows():
        label = None
        for cls in classes:
            if getattr(row, cls, 0.0) == 1.0:
                label = cls
                break
        
        if label:
            mapped = ISIC_MAPPING.get(label, "")
            if mapped in TARGET_CLASSES:
                processed.append({
                    "image_id": row["image"],
                    "dx": mapped,
                    "dataset": "ISIC_2019"
                })
    
    print(f"ISIC 2019: Processed {len(processed)} matching images.")
    return pd.DataFrame(processed)

def process_pad(pad_csv_path: Path) -> pd.DataFrame:
    if not pad_csv_path.exists():
        print(f"Warning: PAD-UFES-20 CSV not found at {pad_csv_path}. Skipping.")
        return pd.DataFrame()
    
    df = pd.read_csv(pad_csv_path)
    # Columns: img_id, diagnostic
    
    processed = []
    for _, row in df.iterrows():
        label = row.get("diagnostic", "")
        mapped = PAD_MAPPING.get(label, "")
        if mapped in TARGET_CLASSES:
            processed.append({
                "image_id": row["img_id"],
                "dx": mapped,
                "dataset": "PAD-UFES-20"
            })
            
    print(f"PAD-UFES-20: Processed {len(processed)} matching images.")
    return pd.DataFrame(processed)

def main():
    parser = argparse.ArgumentParser(description="Prepare a unified dataset by mapping classes.")
    parser.add_argument("--isic_csv", type=str, default="data/ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--pad_csv", type=str, default="data/pad_ufes_20.csv")
    parser.add_argument("--output", type=str, default="data/merged_dataset.csv")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Processing ISIC 2019...")
    isic_df = process_isic(Path(args.isic_csv))
    
    print("Processing PAD-UFES-20...")
    pad_df = process_pad(Path(args.pad_csv))
    
    merged = pd.concat([isic_df, pad_df], ignore_index=True)
    
    if len(merged) > 0:
        merged.to_csv(out_path, index=False)
        print(f"Successfully saved merged dataset with {len(merged)} records to {out_path}")
        print("\nClass distribution:")
        print(merged["dx"].value_counts())
    else:
        print("\nNo data processed. Please ensure the CSV files exist in the 'data/' folder.")
        print("To download them automatically, you can use the Kaggle API:")
        print("  pip install kaggle")
        print("  kaggle datasets download andrewmvd/isic-2019")
        print("  kaggle datasets download bittlingmayer/pad-ufes-20")

if __name__ == "__main__":
    main()
