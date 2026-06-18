import csv
import os


class CSVLogger:

    def __init__(self, filename="icu_features.csv",label="unknown",subject_id="subject_01"):
        self.label = label
        self.subject_id = subject_id

        self.file_exists = os.path.isfile(filename)

        self.file = open(filename, "a", newline="")

        self.writer = csv.writer(self.file)

        if not self.file_exists:
            self.writer.writerow([
                "timestamp",
                "subject_id",
                "frame",
                "ear",
                "blink_rate",
                "perclos",
                "closure_time",
                "yaw",
                "pitch",
                "gaze_variance",
                "saccade_count",
                "recent_saccades",
                "mean_gaze_speed",
                "max_gaze_speed",
                "eye_contact_ratio",
                "fixation_duration",
                "behavior_state"
                "label"
            ])

    def log(
        self,
        timestamp,
        frame,
        ear,
        blink_rate,
        perclos,
        closure_time,
        yaw,
        pitch,
        gaze_variance,
        saccade_count,
        recent_saccades,
        mean_gaze_speed,
        max_gaze_speed,
        eye_contact_ratio,
        fixation_duration,
        behavior_state
         
        
    ):

        self.writer.writerow([
            timestamp,
            self.subject_id,
            frame,
            ear,
            blink_rate,
            perclos,
            closure_time,
            yaw,
            pitch,
            gaze_variance,
            saccade_count,
            recent_saccades,
            mean_gaze_speed,
            max_gaze_speed,
            eye_contact_ratio,
            fixation_duration,
            behavior_state,
            self.label
        ])

        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()