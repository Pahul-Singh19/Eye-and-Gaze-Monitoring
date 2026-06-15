class BlinkDetector:

    def __init__(
        self,
        ear_threshold=0.15,
        min_closed_frames=3
    ):

        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames

        self.closed_frames = 0
        self.blink_count = 0

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

            self.closed_frames = 0

        return {
            "eye_closed": eye_closed,
            "blink_count": self.blink_count
        }