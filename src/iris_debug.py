import cv2
import mediapipe as mp

from eye_cropper import get_eye_features

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                features = get_eye_features(
                    face_landmarks,
                    w,
                    h
                )

                # -----------------------
                # Eye contour landmarks
                # -----------------------

                for x, y in features["eye_points"]:

                    cv2.circle(
                        frame,
                        (x, y),
                        2,
                        (0, 255, 0),
                        -1
                    )

                # -----------------------
                # Iris landmarks
                # -----------------------

                for x, y in features["iris_points"]:

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 0, 255),
                        -1
                    )

                # -----------------------
                # Iris center
                # -----------------------

                iris_x, iris_y = features["iris_center"]

                cv2.circle(
                    frame,
                    (iris_x, iris_y),
                    6,
                    (255, 0, 255),
                    -1
                )

                # -----------------------
                # Eye boundaries
                # -----------------------

                eye_left = features["eye_left"]
                eye_right = features["eye_right"]

                cv2.line(
                    frame,
                    (eye_left, iris_y),
                    (eye_left, iris_y - 20),
                    (0, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (eye_right, iris_y),
                    (eye_right, iris_y - 20),
                    (0, 255, 255),
                    2
                )

                # -----------------------
                # Debug Information
                # -----------------------

                cv2.putText(
                    frame,
                    f"Eye Width: {features['eye_width']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"RelX: {features['rel_x']:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Iris: ({iris_x},{iris_y})",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        cv2.imshow("Iris Debug", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()