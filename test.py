

import cv2
from init import mediapipe_detection, draw_styled_landmarks, extract_keypoints, parameter
import mediapipe as mp
import numpy as np
from creatmodel import create_model
from PIL import ImageFont, ImageDraw, Image
actions = parameter["actions"]
model = create_model(actions)
model.load_weights('action.h5')
mp_holistic = mp.solutions.holistic  # nhận diện toàn cơ thể
mp_drawing = mp.solutions.drawing_utils

font_path = './font/aachenb.ttf'
font_size = 24
font = ImageFont.truetype(font_path, font_size)

sequence = []
sentence = []
threshold = 0.8
FPS = parameter["FPS"]
cap = cv2.VideoCapture(0)

WIDTH = int(cap.get(3))
HIGHT = int(cap.get(4))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        draw_styled_landmarks(mp_holistic, mp_drawing, image, results)

        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-FPS:]

        cv2.rectangle(image, (0, 0), (WIDTH, 40), (245, 117, 16), -1)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        if len(sequence) == FPS:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            # lấy điểm nhỏ nhất trên mặt
            if results.face_landmarks:
                landmarks = results.face_landmarks.landmark
                arr = np.array([[landmark.x, landmark.y]
                               for landmark in landmarks])
                # print(len(landmarks))
                max_x = np.max(arr[:, 0])
                min_y = np.min(arr[:, 1])
                draw.text((max_x*WIDTH, min_y*HIGHT),
                          actions[np.argmax(res)], font=font, fill=(255, 255, 0))
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 3:
                sentence = sentence[-3:]

        draw.text((30, 8), '   '.join(sentence),
                  font=font, fill=(255, 255, 255))
        image = np.array(pil_image)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
