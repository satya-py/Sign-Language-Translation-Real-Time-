"""Live ISL letter/digit recognition using the already-trained action.h5 model.

This is just your notebook's cells 25 + 61 + 62 pulled into one runnable script,
with no retraining and no data collection - it loads the saved model and predicts
live off the webcam.

    python action_live.py
    python action_live.py --model outputs/action.h5
    python action_live.py --threshold 0.6 --history 15

Requirements: tensorflow, opencv-python, mediapipe, numpy
"""

import argparse

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Must match the training order in the notebook (cell 22) EXACTLY: digits then
# letters. The model's output index i means actions[i] - if this list doesn't
# match what label_map used at train time, every prediction is mislabeled even
# though the model itself is fine.
ACTIONS = np.array(
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_hand_keypoints(results):
    """21 landmarks * 3 (x,y,z) * 2 hands = 126 features. Matches the model's
    input_shape=(126,) from training - do not swap in the pose+face+hands
    extractor from earlier in the notebook, its output shape is different."""
    left = np.zeros(21 * 3)
    right = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Left':
                left = coords
            else:
                right = coords
    return np.concatenate([left, right])


def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
            if handedness.classification[0].label == 'Left':
                landmark_spec = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
                connection_spec = mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            else:
                landmark_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)
                connection_spec = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_spec, connection_spec)


def prob_viz(res, actions, input_frame, colors, top_n=5):
    output_frame = input_frame.copy()
    top_indices = np.argsort(res)[-top_n:][::-1]
    for i, num in enumerate(top_indices):
        prob = res[num]
        cv2.rectangle(output_frame, (0, 60 + i * 40), (int(prob * 100), 90 + i * 40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob:.2f}', (0, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="action.h5", help="path to the trained .h5 file")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="minimum probability to accept a prediction")
    p.add_argument("--history", type=int, default=10,
                   help="how many recent frames must agree before a letter is accepted")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    print(f"loading {args.model} ...")
    model = load_model(args.model)

    import matplotlib.pyplot as plt
    colors = [tuple(int(c * 255) for c in plt.cm.hsv(i / len(ACTIONS))[:3]) for i in range(len(ACTIONS))]

    sentence = []
    predictions = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"camera {args.camera} did not open")

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)

            keypoints = extract_hand_keypoints(results)
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))

            # Only accept a prediction once the last `history` frames agree AND
            # its probability clears the threshold. Without this, single noisy
            # frames spam the sentence with wrong/duplicate letters.
            recent = predictions[-args.history:]
            if len(recent) == args.history and np.unique(recent).size == 1 \
                    and recent[0] == np.argmax(res):
                if res[np.argmax(res)] > args.threshold:
                    letter = ACTIONS[np.argmax(res)]
                    if not sentence or letter != sentence[-1]:
                        sentence.append(letter)

            if len(sentence) > 8:
                sentence = sentence[-8:]

            image = prob_viz(res, ACTIONS, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "press 'q' to quit, 'c' to clear", (3, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow('ISL Live', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()