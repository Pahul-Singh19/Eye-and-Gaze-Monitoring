import pandas as pd
import glob
import os

DATA_DIR = "data"

all_data = []

csv_files = glob.glob(
    os.path.join(DATA_DIR, "*.csv")
)

print("Found files:")
for f in csv_files:
    print(f)

for file in csv_files:

    df = pd.read_csv(file)

    all_data.append(df)

combined = pd.concat(all_data)

for label in combined["label"].unique():

    subset = combined[
        combined["label"] == label
    ]

    print("\n====================")
    print(label.upper())
    print("====================")

    print("Samples:", len(subset))

    print(
        "Mean EAR:",
        round(subset["ear"].mean(), 4)
    )

    print(
        "Mean PERCLOS:",
        round(subset["perclos"].mean(), 2)
    )

    print(
        "Mean Closure:",
        round(subset["closure_time"].mean(), 2)
    )

    print(
        "Mean Variance:",
        round(
            subset["gaze_variance"].mean(),
            5
        )
    )

    print(
        "Mean Saccades:",
        round(
            subset["saccade_count"].mean(),
            2
        )
    )
    print(
    "EAR Range:",
    round(subset["ear"].min(),3),
    "->",
    round(subset["ear"].max(),3)
    )

    print(
        "Variance Range:",
        round(subset["gaze_variance"].min(),5),
        "->",
        round(subset["gaze_variance"].max(),5)
    )