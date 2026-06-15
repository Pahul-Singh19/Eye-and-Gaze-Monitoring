class EyeClosureTracker:

    def __init__(self):
        self.current_closed_frames = 0

    def update(self, eye_closed):

        if eye_closed:
            self.current_closed_frames += 1
        else:
            self.current_closed_frames = 0

        return self.current_closed_frames