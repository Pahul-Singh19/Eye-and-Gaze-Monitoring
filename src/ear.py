import numpy as np


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_single_eye_ear(p1, p2, p3, p4, p5, p6):

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_ear(face_landmarks, width, height):

    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return (
            int(lm.x * width),
            int(lm.y * height)
        )

    left_ear = compute_single_eye_ear(
        get_point(33),
        get_point(160),
        get_point(158),
        get_point(133),
        get_point(153),
        get_point(144)
    )

    right_ear = compute_single_eye_ear(
        get_point(362),
        get_point(385),
        get_point(387),
        get_point(263),
        get_point(373),
        get_point(380)
    )

    avg_ear = (left_ear + right_ear) / 2.0

    return left_ear, right_ear, avg_ear