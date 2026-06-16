import pathlib
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from l2cs import Pipeline, render

CWD = pathlib.Path(__file__).parent.resolve()
cudnn.enabled = True

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Running - press q to quit.")

with torch.no_grad():
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        start_fps = time.time()
        results = gaze_pipeline.step(frame)
        frame = render(frame, results)

        fps = 1.0 / (time.time() - start_fps)
        cv2.putText(frame, f'FPS: {fps:.1f}',
                    (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('L2CS-Net Gaze Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()