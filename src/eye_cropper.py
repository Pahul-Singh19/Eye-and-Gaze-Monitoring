from eye_landmarks import extract_eye_landmarks


def get_eye_features(face_landmarks, width, height):

    eye_points, iris_points = extract_eye_landmarks(
        face_landmarks,
        width,
        height
    )

    eye_x = [p[0] for p in eye_points]

    eye_left = min(eye_x)
    eye_right = max(eye_x)

    eye_width = eye_right - eye_left

    iris_x = int(
        sum(p[0] for p in iris_points)
        / len(iris_points)
    )

    iris_y = int(
        sum(p[1] for p in iris_points)
        / len(iris_points)
    )

    if eye_width > 0:
        rel_x = (iris_x - eye_left) / eye_width
    else:
        rel_x = 0

    return {
        "eye_points": eye_points,
        "iris_points": iris_points,
        "iris_center": (iris_x, iris_y),
        "eye_left": eye_left,
        "eye_right": eye_right,
        "eye_width": eye_width,
        "rel_x": rel_x
    }