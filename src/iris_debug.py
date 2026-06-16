import cv2
import mediapipe as mp
import time

from eye_cropper import get_eye_features
from ear import compute_ear
from blink_detector import BlinkDetector
from perclos import Perclos
from eye_closure_tracker import EyeClosureTracker
from gaze_estimator import GazeEstimator
from gaze_tracker import GazeTracker

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)



with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    blink_detector = BlinkDetector(
        ear_threshold=0.15,
        min_closed_frames=3
    ) 
    perclos_tracker = Perclos()
    closure_tracker=EyeClosureTracker()
    gaze = GazeEstimator()
    gaze_tracker=GazeTracker()
    frame_count = 0
    last_yaw = None
    last_pitch = None
    prev_time = time.time()


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
                
                left_ear, right_ear, ear = compute_ear(
                    face_landmarks,
                    w,
                    h
                )
                
                
                blink_data = blink_detector.update(
                    left_ear,
                    right_ear
                )


                perclos = perclos_tracker.update(blink_data["eye_closed"])
                closed_frames = closure_tracker.update(
                    blink_data["eye_closed"]
                )

                closed_time = closed_frames / 30.0
                            


                features = get_eye_features(
                    face_landmarks,
                    w,
                    h
                )
                xs = [
                    int(lm.x * w)
                    for lm in face_landmarks.landmark
                ]

                ys = [
                    int(lm.y * h)
                    for lm in face_landmarks.landmark
                ]

                padding = 20

                x_min = max(0, min(xs) - padding)
                y_min = max(0, min(ys) - padding)

                x_max = min(w, max(xs) + padding)
                y_max = min(h, max(ys) + padding)

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 0),
                    2
                )
                face_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]
                cv2.imshow("Face Crop", face_crop)
                frame_count += 1

                if frame_count % 3 == 0:
                

                    last_yaw, last_pitch = gaze.estimate(face_crop)
                yaw =last_yaw
                pitch = last_pitch
                gaze_data = None
                
                
                if yaw is not None:
                    gaze_data = gaze_tracker.update(
                    yaw,
                    pitch
                )
                
                    
                    cv2.putText(
                        frame,
                        f"Yaw:{yaw*57.3:.1f} Pitch:{pitch*57.3:.1f}",
                        (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                if gaze_data is not None:
                    cv2.putText(
                    frame,
                    f"Yaw Var:{gaze_data['yaw_variance']:.4f}",
                    (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                    cv2.putText(
                        frame,
                        f"Pitch Var:{gaze_data['pitch_variance']:.4f}",
                        (10, 440),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
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

                cv2.putText(
                    frame,
                    f"L-EAR: {left_ear:.3f}",
                    (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"R-EAR: {right_ear:.3f}",
                    (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"AVG-EAR: {ear:.3f}",
                    (10,180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Blinks: {blink_data['blink_count']}",
                    (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2
                )
                status = (
                    "CLOSED"
                    if blink_data["eye_closed"]
                    else "OPEN"
                )

                cv2.putText(
                    frame,
                    f"Eye: {status}",
                    (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )
                
                cv2.putText(
                    frame,
                    f"PERCLOS: {perclos:.2f}%",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Closed: {closed_time:.1f}s",
                    (10,240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )
        current_time = time.time()

        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )




        cv2.imshow("Iris Debug", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()