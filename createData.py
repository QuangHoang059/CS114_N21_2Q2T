import cv2
import numpy as np
import os
import mediapipe as mp
from init import mediapipe_detection, draw_styled_landmarks, extract_keypoints, parameter
mp_holistic = mp.solutions.holistic  # nhận diện toàn cơ thể
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = parameter["DATA_PATH"]
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

actions = parameter["actions"]
no_sequences = parameter["no_sequences"]  # số lần lấy data
FPS = parameter["FPS"]  # só frame lấy được

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Collect action {action}', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                # Lưu ảnh
                break

        pathAction = os.path.join(DATA_PATH, action)
        if not os.path.exists(pathAction):
            os.makedirs(pathAction)
        for sequence in range(no_sequences):
            if not os.path.exists(os.path.join(pathAction, str(sequence))):
                os.makedirs(os.path.join(pathAction, str(sequence)))
            
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            cv2.putText(image, 'PREPAIR COLLECTION', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(2000)
            
            for frame_num in range(FPS):

                ret, frame = cap.read()
                # frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)

                draw_styled_landmarks(mp_holistic, mp_drawing, image, results)

                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    pathAction, str(sequence), str(frame_num))

                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    exit(0)

    cap.release()
    cv2.destroyAllWindows()
