## IT WILL WORK, right now it is in developing 
## left,right click, scroll, v shape, volume control, play/ pause, drag/drop  these are working
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyautogui
import subprocess

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands for two hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Landmarks for each detected hand
        self.landmarks = {}
        self.handedness = {}
        self.hand_present = False
        
        # Gesture state tracking for transitions
        self.prev_gestures = {}
        self.gesture_history = {'Left': [], 'Right': []}
        self.history_length = 3

    def process_frame(self, frame):
        # Reset per-frame data
        self.landmarks = {}
        self.handedness = {}
        self.hand_present = False

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            self.hand_present = True
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left or Right)
                hand_label = results.multi_handedness[i].classification[0].label
                
                # Store landmarks and handedness
                self.landmarks[hand_label] = hand_landmarks
                self.handedness[hand_label] = results.multi_handedness[i]
                
                # Draw hand landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame

    def get_landmark_position(self, hand_label, landmark_id):
        if hand_label not in self.landmarks:
            return None
        
        landmark = self.landmarks[hand_label].landmark[landmark_id]
        return (landmark.x, landmark.y, landmark.z)

    def get_distance(self, hand_label, landmark_id1, landmark_id2):
        pos1 = self.get_landmark_position(hand_label, landmark_id1)
        pos2 = self.get_landmark_position(hand_label, landmark_id2)
        
        if not pos1 or not pos2:
            return None
        
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def is_finger_extended(self, hand_label, finger_tip_id, finger_pip_id):
        # A finger is extended if its tip is further from the wrist than its PIP joint
        if hand_label not in self.landmarks:
            return False
            
        wrist_pos = self.get_landmark_position(hand_label, 0)
        tip_pos = self.get_landmark_position(hand_label, finger_tip_id)
        pip_pos = self.get_landmark_position(hand_label, finger_pip_id)

        if not all([wrist_pos, tip_pos, pip_pos]):
            return False

        dist_tip_wrist = math.sqrt((tip_pos[0] - wrist_pos[0])**2 + (tip_pos[1] - wrist_pos[1])**2)
        dist_pip_wrist = math.sqrt((pip_pos[0] - wrist_pos[0])**2 + (pip_pos[1] - wrist_pos[1])**2)
        
        return dist_tip_wrist > dist_pip_wrist
        
    def recognize_all_gestures(self):
        recognized_gestures = {}
        for hand_label in self.landmarks.keys():
            if hand_label == 'Right':
                gesture = self.recognize_right_hand_gesture()
            else: # Left Hand
                gesture = self.recognize_left_hand_gesture()
            
            # Smooth gesture
            if len(self.gesture_history[hand_label]) >= self.history_length:
                self.gesture_history[hand_label].pop(0)
            self.gesture_history[hand_label].append(gesture)
            
            # Use the most common gesture in history for stability
            if self.gesture_history[hand_label]:
                smoothed_gesture = max(set(self.gesture_history[hand_label]), key=self.gesture_history[hand_label].count)
                recognized_gestures[hand_label] = smoothed_gesture

        # Handle transitions (e.g., Open Palm -> Fist)
        for hand_label, current_gesture in recognized_gestures.items():
            prev_gesture = self.prev_gestures.get(hand_label)
            if prev_gesture == 'Open Palm' and current_gesture == 'Fist':
                recognized_gestures[hand_label] = 'Palm to Fist Transition'
            self.prev_gestures[hand_label] = current_gesture # Update previous gesture
        
        return recognized_gestures
        
    def recognize_right_hand_gesture(self):
        hand_label = 'Right'
        if hand_label not in self.landmarks:
            return "No Hand"

        # Finger extension states
        thumb_extended = self.is_finger_extended(hand_label, 4, 3)
        index_extended = self.is_finger_extended(hand_label, 8, 6)
        middle_extended = self.is_finger_extended(hand_label, 12, 10)
        ring_extended = self.is_finger_extended(hand_label, 16, 14)
        pinky_extended = self.is_finger_extended(hand_label, 20, 18)
        
        all_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        
        # -- GESTURE LOGIC --
        
        # Fist
        if not any(all_fingers):
            return "Fist"
        
        # Open Palm
        if all(all_fingers):
            return "Open Palm"

        # Minimize (All fingers closed)
        if not any(all_fingers):
            return "Minimize"

        # Drag & Drop (Pinch with ring/pinky closed)
        pinch_dist = self.get_distance(hand_label, 4, 8)
        if pinch_dist and pinch_dist < 0.04 and not ring_extended and not pinky_extended:
            return "Drag & Drop"
            
        # Left Click (Index open, thumb touching its side)
        thumb_index_side_dist = self.get_distance(hand_label, 4, 7)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended and thumb_index_side_dist and thumb_index_side_dist < 0.07:
            return "Left Click"
            
        # Right Click (Index/Middle open, thumb near ring)
        thumb_ring_dist = self.get_distance(hand_label, 4, 16)
        if index_extended and middle_extended and not ring_extended and not pinky_extended and thumb_ring_dist and thumb_ring_dist < 0.1:
            return "Right Click"
            
        # Scroll (Index straight, thumb near middle tip)
        thumb_index_dist_up = self.get_distance(hand_label, 4, 10)
        thumb_index_dist_down = self.get_distance(hand_label, 4, 12)
        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            if thumb_index_dist_up and thumb_index_dist_up < 0.08:
                return "Scroll Up"
            elif thumb_index_dist_down and thumb_index_dist_down < 0.08:
                return "Scroll Down"

        
        # Mouse Move (Index/Middle open and together)
        index_middle_dist = self.get_distance(hand_label, 8, 12)
        if index_extended and middle_extended and not ring_extended and not pinky_extended and index_middle_dist and index_middle_dist < 0.08:
            return "Mouse Move"
        
        # Tab Switching (3 fingers open)
        if index_extended and middle_extended and ring_extended and not pinky_extended:
            return "Tab Switch"

        # Slides Navigation (4 fingers open)
        if index_extended and middle_extended and ring_extended and pinky_extended and not thumb_extended:
            return "Slides Nav"

        # Next Track (Thumb extended right)
        thumb_tip_pos = self.get_landmark_position(hand_label, 4)
        thumb_mcp_pos = self.get_landmark_position(hand_label, 2)
        if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_tip_pos[0] < thumb_mcp_pos[0]: # Flipped view
            return "Next Track"
            
        # Previous Track (Pinky extended left)
        pinky_tip_pos = self.get_landmark_position(hand_label, 20)
        pinky_mcp_pos = self.get_landmark_position(hand_label, 17)
        if pinky_extended and not any([thumb_extended, index_extended, middle_extended, ring_extended]) and pinky_tip_pos[0] > pinky_mcp_pos[0]: # Flipped view
            return "Previous Track"

        return "Unknown"
        
    def recognize_left_hand_gesture(self):
        hand_label = 'Left'
        if hand_label not in self.landmarks:
            return "No Hand"

        # Finger extension states
        thumb_extended = self.is_finger_extended(hand_label, 4, 3)
        index_extended = self.is_finger_extended(hand_label, 8, 6)
        middle_extended = self.is_finger_extended(hand_label, 12, 10)
        ring_extended = self.is_finger_extended(hand_label, 16, 14)
        pinky_extended = self.is_finger_extended(hand_label, 20, 18)
        
        all_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

        # Fist
        if not any(all_fingers):
            return "Fist"
        
        # Open Palm
        if all(all_fingers):
            return "Open Palm"

        # Volume Control (Pinch)
        pinch_dist = self.get_distance(hand_label, 4, 8)
        if pinch_dist is not None:
             # A loose pinch is fine for volume control
            if index_extended and thumb_extended and not any([middle_extended, ring_extended, pinky_extended]):
                 return "Volume Control"
            
        # V-Shape Gesture (Left Hand)
        index_middle_dist = self.get_distance(hand_label, 8, 12)  # index tip to middle tip
        if index_extended and middle_extended and not ring_extended and not pinky_extended and index_middle_dist and index_middle_dist >= 0.08:
            return "V Shape"


        return "Unknown"


class HandController:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Action cooldowns
        self.last_action_time = {}
        self.cooldown_period = 0.5 # Default cooldown
        
        # State variables
        self.is_dragging = False
        self.prev_hand_pos = {'Right': None, 'Left': None}
        self.scroll_origin_y = None

    def can_perform_action(self, action_name, cooldown=None):
        cooldown = cooldown if cooldown is not None else self.cooldown_period
        current_time = time.time()
        if action_name in self.last_action_time:
            if current_time - self.last_action_time[action_name] < cooldown:
                return False
        
        self.last_action_time[action_name] = current_time
        return True
    
    def handle_gestures(self, frame):
        gestures = self.recognizer.recognize_all_gestures()
        
        # Display Gestures
        cv2.putText(frame, f"Right: {gestures.get('Right', 'N/A')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Left: {gestures.get('Left', 'N/A')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Handle Right Hand
        if 'Right' in gestures and 'Right' in self.recognizer.landmarks:
            self.handle_right_hand(gestures['Right'], frame)
            
        # Handle Left Hand
        if 'Left' in gestures and 'Left' in self.recognizer.landmarks:
            self.handle_left_hand(gestures['Left'], frame)
        
        # Reset drag state if the gesture is lost
        if gestures.get('Right') != 'Drag & Drop' and self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
            print("Drag released")
            
        return frame

    def handle_right_hand(self, gesture, frame):
        hand_label = 'Right'
        landmarks = self.recognizer.landmarks[hand_label]
        
        if gesture == "Mouse Move":
            index_tip = landmarks.landmark[8]
            target_x = np.interp(index_tip.x, [0.2, 0.8], [0, self.screen_width])
            target_y = np.interp(index_tip.y, [0.2, 0.8], [0, self.screen_height])
            pyautogui.moveTo(target_x, target_y, duration=0.1)
        
        elif gesture == "Left Click" and self.can_perform_action("left_click", 0.3):
            pyautogui.click()
            cv2.putText(frame, "Action: Left Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elif gesture == "Right Click" and self.can_perform_action("right_click"):
            pyautogui.rightClick()
            cv2.putText(frame, "Action: Right Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        
        elif gesture in ["Scroll Up", "Scroll Down"]:
            # Determine reference landmark based on gesture
            if gesture == "Scroll Up":
                ref_y = landmarks.landmark[10].y  # landmark 10 for scroll up
            else:
                ref_y = landmarks.landmark[12].y  # landmark 12 for scroll down

            if self.scroll_origin_y is None:
                self.scroll_origin_y = ref_y

            y_diff = ref_y - self.scroll_origin_y
            if abs(y_diff) > 0.03:
                scroll_amount = 1 if gesture == "Scroll Up" else -1
                pyautogui.scroll(scroll_amount * 20)  # Scroll 20 units
                self.scroll_origin_y = ref_y  # Reset origin
        else:
            self.scroll_origin_y = None  # Reset when not scrolling


        if gesture == "Drag & Drop":
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                print("Drag initiated")
            index_tip = landmarks.landmark[8]
            target_x = np.interp(index_tip.x, [0.2, 0.8], [0, self.screen_width])
            target_y = np.interp(index_tip.y, [0.2, 0.8], [0, self.screen_height])
            pyautogui.moveTo(target_x, target_y, duration=0.1)

        # elif gesture == "Palm to Fist Transition" and self.can_perform_action("minimize"):
        #     pyautogui.hotkey('command', 'm')
        #     cv2.putText(frame, "Action: Minimize Window", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif gesture == "Fingers Touching" and self.can_perform_action("minimize"):
            pyautogui.hotkey('command', 'm')  # macOS minimize
            # pyautogui.hotkey('win', 'down')  # Windows minimize (uncomment if needed)
            cv2.putText(frame, "Action: Minimize Window", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

       
        elif gesture in ["Tab Switch", "Slides Nav"]:
            wrist_pos_x = landmarks.landmark[0].x
            if self.prev_hand_pos[hand_label] is not None:
                x_diff = wrist_pos_x - self.prev_hand_pos[hand_label]
                if abs(x_diff) > 0.05 and self.can_perform_action(gesture, 0.8):
                    key_next = 'right' if gesture == "Slides Nav" else ('command', 'option', 'right')
                    key_prev = 'left' if gesture == "Slides Nav" else ('command', 'option', 'left')
                    if x_diff < 0: # Swipe L->R in mirrored view
                        pyautogui.hotkey(*key_prev)
                        action_text = "Prev Tab/Slide"
                    else: # Swipe R->L
                        pyautogui.hotkey(*key_next)
                        action_text = "Next Tab/Slide"
                    cv2.putText(frame, f"Action: {action_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.prev_hand_pos[hand_label] = wrist_pos_x
        else:
            self.prev_hand_pos[hand_label] = None

        # if gesture == "V Shape" and self.can_perform_action("play_pause", 1.0):
        #     pyautogui.press('space')
        #     cv2.putText(frame, "Action: Play/Pause", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if gesture == "Next Track" and self.can_perform_action("next_track"):
            pyautogui.press('nexttrack')
            cv2.putText(frame, "Action: Next Track", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif gesture == "Previous Track" and self.can_perform_action("prev_track"):
            pyautogui.press('prevtrack')
            cv2.putText(frame, "Action: Previous Track", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def handle_left_hand(self, gesture, frame):
        hand_label = 'Left'

        if gesture == "Palm to Fist Transition" and self.can_perform_action("close_window"):
            pyautogui.hotkey('command', 'w')
            cv2.putText(frame, "Action: Close Window/Tab", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif gesture == "Volume Control":
            # Check finger states
            thumb_extended = self.recognizer.is_finger_extended(hand_label, 4, 3)
            index_extended = self.recognizer.is_finger_extended(hand_label, 8, 6)
            middle_extended = self.recognizer.is_finger_extended(hand_label, 12, 10)
            ring_extended = self.recognizer.is_finger_extended(hand_label, 16, 14)
            pinky_extended = self.recognizer.is_finger_extended(hand_label, 20, 18)

            # Volume control only if middle, ring, pinky are all closed
            if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
                pinch_dist = self.recognizer.get_distance(hand_label, 4, 8)
                if pinch_dist is not None:
                    # Map pinch distance [0.02, 0.2] to volume [0, 100]
                    volume_level = int(np.interp(pinch_dist, [0.02, 0.2], [0, 100]))
                    volume_level = max(0, min(100, volume_level))  # Clamp value
                    
                    # Set volume using osascript (macOS specific)
                    subprocess.run(['osascript', '-e', f'set volume output volume {volume_level}'], capture_output=True)
                    cv2.putText(frame, f"Volume: {volume_level}%", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elif gesture == "V Shape" and self.can_perform_action("play_pause", 1.0):
            pyautogui.press('space')
            cv2.putText(frame, "Action: Play/Pause", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    recognizer = HandGestureRecognizer()
    controller = HandController(recognizer)
    
    cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_NORMAL)
    
    print("=== Hand Gesture Control System ===")
    print("Show your hands to the camera.")
    print("Right Hand: Mouse, Clicks, Media, Windows")
    print("Left Hand: Volume, Close Window")
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frame = recognizer.process_frame(frame)
        frame = controller.handle_gestures(frame)
        
        cv2.imshow('Hand Gesture Control', frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






