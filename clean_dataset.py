import pandas as pd
import glob
import os

DATA_DIR = "C:\ICU monitoring\EYEGAZE\data"

for file in glob.glob(os.path.join(DATA_DIR, "*.csv")):

    print(f"Cleaning {file}")

    df = pd.read_csv(file)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Standardize labels
    if "label" in df.columns:

        df["label"] = (
            df["label"]
            .replace("fixed", "fixation")
            .replace("eyeclosed", "eye_closed")
        )

    # Remove rows with missing label
    df = df.dropna(subset=["label"])

    df.to_csv(file, index=False)

    print(f"Rows remaining: {len(df)}")

print("\nAll datasets cleaned.")