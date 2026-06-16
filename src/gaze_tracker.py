from collections import deque
import numpy as np


class GazeTracker:

    def __init__(self, window_size=150):

        self.yaw_history = deque(maxlen=window_size)
        self.pitch_history = deque(maxlen=window_size)

    def update(self, yaw, pitch):

        if yaw is None:
            return None

        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        yaw_var = np.var(self.yaw_history)
        pitch_var = np.var(self.pitch_history)

        return {
            "yaw_variance": yaw_var,
            "pitch_variance": pitch_var
        }