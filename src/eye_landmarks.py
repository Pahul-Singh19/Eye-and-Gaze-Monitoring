import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

LEFT_IRIS = [468, 469, 470, 471, 472]

RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

RIGHT_IRIS = [473, 474, 475, 476, 477]


def extract_eye_landmarks(face_landmarks, width, height):

    eye_points = []

    for idx in LEFT_EYE:

        lm = face_landmarks.landmark[idx]

        x = int(lm.x * width)
        y = int(lm.y * height)

        eye_points.append((x, y))

    iris_points = []

    for idx in LEFT_IRIS:

        lm = face_landmarks.landmark[idx]

        x = int(lm.x * width)
        y = int(lm.y * height)

        iris_points.append((x, y))

    return eye_points, iris_points