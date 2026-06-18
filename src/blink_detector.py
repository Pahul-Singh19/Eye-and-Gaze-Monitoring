from collections import deque
import time


class BlinkDetector:

    def __init__(
        self,
        ear_threshold=0.16,
        min_closed_frames=3
    ):

        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames

        self.closed_frames = 0
        self.blink_count = 0

        # Stores timestamps of recent blinks
        self.blink_times = deque()

    def update(self, left_ear, right_ear):

        eye_closed = False

        if (
            left_ear < self.ear_threshold
            and
            right_ear < self.ear_threshold
        ):

            self.closed_frames += 1
            eye_closed = True

        else:

            if self.closed_frames >= self.min_closed_frames:

                self.blink_count += 1

                # Store blink timestamp
                self.blink_times.append(
                    time.time()
                )

            self.closed_frames = 0

        # -------------------------
        # Blink Rate (last 60 sec)
        # -------------------------

        current_time = time.time()

        while (
            len(self.blink_times) > 0
            and
            current_time - self.blink_times[0] > 60
        ):
            self.blink_times.popleft()

        blink_rate = len(
            self.blink_times
        )

        return {
            "eye_closed": eye_closed,
            "blink_count": self.blink_count,
            "blink_rate": blink_rate
        }