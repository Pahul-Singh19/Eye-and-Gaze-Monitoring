from collections import deque


class Perclos:

    def __init__(self, window_size=1800):
        # 60 seconds at ~30 FPS
        self.window = deque(maxlen=window_size)

    def update(self, eye_closed):

        self.window.append(
            1 if eye_closed else 0
        )

        return (
            sum(self.window)
            / len(self.window)
        ) * 100