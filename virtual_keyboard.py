import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)
frame_width = 1280
frame_height = 720
cap.set(3, frame_width)
cap.set(4, frame_height)

# Keyboard layout
keyboard_layout = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'BACK'],
    ['TAB', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']', '\\'],
    ['CAPS', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'ENTER'],
    ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', 'SHIFT'],
    ['CTRL', 'ALT', 'SPACE', 'ALT', 'CTRL']
]

# States
typed_text = ""
key_boxes = []
pressed_fingers = {}
caps_lock = False
shift_active = False

fingertips = [4, 8, 12, 16, 20]
fingerjoints = [3, 6, 10, 14, 18]


def draw_keyboard(img):
    key_boxes.clear()
    key_h = 60
    row_gap = 15  # Increased spacing between rows
    key_gap = 10  # Increased spacing between keys
    y_start = 220  # Moved keyboard slightly up

    for row_idx, row in enumerate(keyboard_layout):
        total_width = sum([300 if k == 'SPACE' else 100 if k in ['SHIFT', 'ENTER', 'TAB', 'CAPS', 'BACK'] else 80 if k in ['CTRL', 'ALT'] else 60 for k in row]) + key_gap * (len(row) - 1)
        x_start = (frame_width - total_width) // 2
        x = x_start

        for key in row:
            if key == 'SPACE':
                w = 300
            elif key in ['SHIFT', 'ENTER', 'TAB', 'CAPS', 'BACK']:
                w = 100
            elif key in ['CTRL', 'ALT']:
                w = 80
            else:
                w = 60
            box = (x, y_start + row_idx * (key_h + row_gap), w, key_h)
            key_boxes.append((box, key))
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (50, 50, 50), -1)
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 255), 2)
            text = key if key != "SPACE" else ""
            font_scale = 0.8
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            cx = box[0] + (box[2] - size[0]) // 2
            cy = box[1] + (box[3] + size[1]) // 2
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            x += w + key_gap


def detect_key(finger_tip, frame):
    global typed_text, caps_lock, shift_active
    x, y = finger_tip
    for box, key in key_boxes:
        x1, y1, w, h = box
        if x1 < x < x1 + w and y1 < y < y1 + h:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

            if key == "SPACE":
                typed_text += " "
                pyautogui.press("space")
            elif key == "BACK":
                typed_text = typed_text[:-1]
                pyautogui.press("backspace")
            elif key == "ENTER":
                typed_text += "\n"
                pyautogui.press("enter")
            elif key == "CAPS":
                caps_lock = not caps_lock
            elif key == "SHIFT":
                shift_active = not shift_active
            elif key in ['CTRL', 'ALT', 'TAB']:
                pyautogui.press(key.lower())
            else:
                char = key
                if shift_active ^ caps_lock:
                    char = char.upper()
                else:
                    char = char.lower()
                typed_text += char
                pyautogui.press(char)
            return True
    return False


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    draw_keyboard(img)

    # Display typed text
    cv2.rectangle(img, (60, 20), (1220, 120), (0, 0, 0), -1)
    lines = typed_text.split('\n')[-2:]
    for i, line in enumerate(lines):
        cv2.putText(img, line[-80:], (70, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_id, handLms in enumerate(results.multi_hand_landmarks):
            lmList = []
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            for i in range(5):
                tip_id = fingertips[i]
                joint_id = fingerjoints[i]

                if tip_id < len(lmList) and joint_id < len(lmList):
                    tip = lmList[tip_id]
                    joint = lmList[joint_id]
                    cv2.circle(img, tip, 10, (0, 255, 255), -1)

                    finger_key = f"{hand_id}_{tip_id}"

                    if tip[1] - joint[1] > 25:
                        if not pressed_fingers.get(finger_key, False):
                            if detect_key(tip, img):
                                pressed_fingers[finger_key] = True
                    else:
                        pressed_fingers[finger_key] = False

    cv2.imshow("Virtual Keyboard - Indian QWERTY", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


