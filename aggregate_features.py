import pandas as pd
import glob

WINDOW_SIZE = 10

all_windows = []

csv_files = glob.glob("data/*.csv")

for file in csv_files:

    df = pd.read_csv(file)

    df = df.dropna(how="all")

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        WINDOW_SIZE
    ):

        window = df.iloc[
            start:start + WINDOW_SIZE
        ]

        saccade_delta = (
            window["saccade_count"].iloc[-1]
            -
            window["saccade_count"].iloc[0]
        )

        feature_row = {

            # -------------------------
            # Eye openness
            # -------------------------

            "mean_ear":
                window["ear"].mean(),

            "std_ear":
                window["ear"].std(),

            # -------------------------
            # Blink activity
            # -------------------------

            "blink_rate":
                window["blink_count"].iloc[-1]
                -
                window["blink_count"].iloc[0],

            # -------------------------
            # Drowsiness
            # -------------------------

            "mean_perclos":
                window["perclos"].mean(),

            "max_perclos":
                window["perclos"].max(),

            "mean_closure_time":
                window["closure_time"].mean(),

            "max_closure_time":
                window["closure_time"].max(),

            # -------------------------
            # Gaze direction
            # -------------------------

            "mean_yaw":
                window["yaw"].mean(),

            "std_yaw":
                window["yaw"].std(),

            "yaw_range":
                window["yaw"].max()
                -
                window["yaw"].min(),

            "mean_pitch":
                window["pitch"].mean(),

            "std_pitch":
                window["pitch"].std(),

            "pitch_range":
                window["pitch"].max()
                -
                window["pitch"].min(),

            # -------------------------
            # Gaze variance
            # -------------------------

            "mean_gaze_variance":
                window["gaze_variance"].mean(),

            "max_gaze_variance":
                window["gaze_variance"].max(),

            "std_gaze_variance":
                window["gaze_variance"].std(),

            # -------------------------
            # Fixation behaviour
            # -------------------------

            "mean_fixation_duration":
                window["fixation_duration"].mean(),

            "max_fixation_duration":
                window["fixation_duration"].max(),

            # -------------------------
            # Recent saccades
            # -------------------------

            "mean_recent_saccades":
                window["recent_saccades"].mean(),

            "max_recent_saccades":
                window["recent_saccades"].max(),

            # -------------------------
            # Gaze speed
            # -------------------------

            "mean_gaze_speed":
                window["mean_gaze_speed"].mean(),

            "max_gaze_speed":
                window["max_gaze_speed"].max(),

            # -------------------------
            # Lifetime saccades
            # -------------------------

            "saccade_delta":
                saccade_delta,

            # -------------------------
            # Label
            # -------------------------

            "label":
                window["label"].mode()[0]
        }

        all_windows.append(feature_row)

aggregated_df = pd.DataFrame(all_windows)

aggregated_df.to_csv(
    "aggregated_dataset.csv",
    index=False
)

print(
    f"Created dataset with "
    f"{len(aggregated_df)} windows"
)

print("\nFirst 5 rows\n")
print(aggregated_df.head())

print("\nClass Distribution\n")
print(
    aggregated_df["label"]
    .value_counts()
)

print("\nColumns:")
print(
    aggregated_df.columns.tolist()
)

print("\nShape:")
print(
    aggregated_df.shape
)

pd.set_option(
    "display.max_columns",
    None
)

print("\nPer-Class Feature Means\n")

print(
    aggregated_df
    .groupby("label")
    .mean(numeric_only=True)
)