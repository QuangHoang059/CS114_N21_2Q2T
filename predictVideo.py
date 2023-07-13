
import cv2
from init import mediapipe_detection, draw_styled_landmarks, extract_keypoints, parameter
import mediapipe as mp
import numpy as np
import os
from creatmodel import create_model
actions = parameter["actions"]
model = create_model(actions)
model.load_weights('action.h5')
mp_holistic = mp.solutions.holistic  # nhận diện toàn cơ thể
mp_drawing = mp.solutions.drawing_utils

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'alpaca.mp4')
FPS = parameter["FPS"]
# video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)


sequence = []
sentence = []
threshold = 0.8

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        draw_styled_landmarks(mp_holistic, mp_drawing, image, results)

        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-FPS:]

        if len(sequence) == FPS:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
