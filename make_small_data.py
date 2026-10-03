import pandas as pd
import os

# Define the paths to your 6 massive files
# (Make sure these paths match exactly where your folders are!)
huge_files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]

print("Shrinking datasets for local testing...")

for file_path in huge_files:
    if os.path.exists(file_path):
        # Create a new name, e.g., train_source1_small.tsv
        new_path = file_path.replace(".tsv", "_small.tsv")
        
        # Read ONLY the first 50,000 rows (super fast and uses almost zero RAM)
        df = pd.read_csv(file_path, sep="\t", nrows=50000, dtype=str)
        
        # Save it to the new small file
        df.to_csv(new_path, sep="\t", index=False)
        print(f"✅ Created: {new_path}")
    else:
        print(f"❌ Could not find: {file_path}")

print("\nDone! Now update your config.py to point to these new '_small' files.")