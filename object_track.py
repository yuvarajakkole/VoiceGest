# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# import pyautogui
# import subprocess
# import screen_brightness_control as sbc

# # Disable PyAutoGUI fail-safe
# pyautogui.FAILSAFE = False

# class HandGestureRecognizer:
#     def __init__(self):
#         # Initialize MediaPipe Hands
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_drawing_styles = mp.solutions.drawing_styles
        
#         # Initialize landmarks and gesture states
#         self.landmarks = None
#         self.hand_present = False
#         self.prev_gesture = None
#         self.gesture_start_time = 0
#         self.gesture_duration = 0
        
#         # Gesture detection thresholds
#         self.pinch_threshold = 0.05
#         self.finger_threshold = 0.4
        
#         # Gesture states
#         self.pinch_state = False
#         self.prev_pinch_state = False
#         self.pinch_start_time = 0
#         self.double_click_threshold = 0.5
#         self.last_pinch_time = 0
        
#         # Gesture history for smoothing
#         self.gesture_history = []
#         self.history_length = 5
        
#         # Previous hand position for movement detection
#         self.prev_hand_center = None
#         self.hand_movement_threshold = 0.05
        
#         # Previous finger positions for swipe detection
#         self.prev_finger_positions = {}
#         self.swipe_threshold = 0.1
        
#         # Circle gesture detection
#         self.circle_points = []
#         self.circle_max_points = 20
#         self.circle_threshold = 0.7
        
#         # Scroll state
#         self.prev_scroll_y = None
#         self.scroll_sensitivity = 10
        
#         # Volume control (macOS specific)
#         self.volume_min = 0
#         self.volume_max = 100
        
#         # Screen dimensions
#         self.screen_width, self.screen_height = pyautogui.size()
        
#         # Mode selection
#         self.modes = ["Basic Control", "App Management", "Media Control", "Volume & Brightness", "Advanced"]
#         self.current_mode = 0
#         self.mode_change_cooldown = 0
        
#     def process_frame(self, frame):
#         # Convert the BGR image to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Process the frame with MediaPipe Hands
#         results = self.hands.process(rgb_frame)
        
#         # Reset hand presence flag
#         self.hand_present = False
        
#         if results.multi_hand_landmarks:
#             self.hand_present = True
#             self.landmarks = results.multi_hand_landmarks[0]
            
#             # Draw hand landmarks on the frame
#             self.mp_drawing.draw_landmarks(
#                 frame,
#                 self.landmarks,
#                 self.mp_hands.HAND_CONNECTIONS,
#                 self.mp_drawing_styles.get_default_hand_landmarks_style(),
#                 self.mp_drawing_styles.get_default_hand_connections_style()
#             )
            
#             # Update gesture history
#             current_gesture = self.recognize_gesture()
#             if len(self.gesture_history) >= self.history_length:
#                 self.gesture_history.pop(0)
#             self.gesture_history.append(current_gesture)
#         else:
#             self.landmarks = None
            
#         return frame
    
#     def get_landmark_position(self, landmark_id):
#         if not self.landmarks:
#             return None
        
#         landmark = self.landmarks.landmark[landmark_id]
#         return (landmark.x, landmark.y, landmark.z)
    
#     def get_distance(self, landmark_id1, landmark_id2):
#         pos1 = self.get_landmark_position(landmark_id1)
#         pos2 = self.get_landmark_position(landmark_id2)
        
#         if not pos1 or not pos2:
#             return None
        
#         return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
#     def get_hand_center(self):
#         if not self.landmarks:
#             return None
        
#         x_sum = 0
#         y_sum = 0
#         for landmark in self.landmarks.landmark:
#             x_sum += landmark.x
#             y_sum += landmark.y
        
#         return (x_sum / 21, y_sum / 21)
    
#     def is_finger_extended(self, finger_tip_id, finger_pip_id, finger_mcp_id):
#         if not self.landmarks:
#             return False
        
#         tip = self.landmarks.landmark[finger_tip_id]
#         pip = self.landmarks.landmark[finger_pip_id]
#         mcp = self.landmarks.landmark[finger_mcp_id]
        
#         # Check if the tip is extended beyond the pip
#         if tip.y < pip.y:
#             return True
#         return False
    
#     def is_thumb_extended(self):
#         if not self.landmarks:
#             return False
        
#         thumb_tip = self.landmarks.landmark[4]
#         thumb_ip = self.landmarks.landmark[3]
#         thumb_mcp = self.landmarks.landmark[2]
        
#         # For thumb, we check if it's extended to the side
#         if thumb_tip.x < thumb_ip.x:
#             return True
#         return False
    
#     def recognize_gesture(self):
#         if not self.landmarks:
#             return "No Hand"
        
#         # Check for basic finger extensions
#         is_thumb_extended = self.is_thumb_extended()
#         is_index_extended = self.is_finger_extended(7, 6, 5)
#         is_middle_extended = self.is_finger_extended(12, 10, 9)
#         is_ring_extended = self.is_finger_extended(16, 14, 13)
#         is_pinky_extended = self.is_finger_extended(20, 18, 17)
        
#         # Get distances for pinch gestures
#         thumb_index_distance = self.get_distance(4, 8)
#         thumb_middle_distance = self.get_distance(4, 12)
        
#         # Get hand center for movement detection
#         hand_center = self.get_hand_center()
        
#         # Basic System Control Gestures
        
#         # Point (Move Mouse) - Index finger extended, others curled
#         if is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             return "Point"
        
#         # Pinch Tap (Left Click) - Thumb and Index close
#         if thumb_index_distance and thumb_index_distance < self.pinch_threshold:
#             # Track pinch state for click detection
#             self.prev_pinch_state = self.pinch_state
#             self.pinch_state = True
            
#             # Detect click (transition from not pinched to pinched)
#             if not self.prev_pinch_state and self.pinch_state:
#                 current_time = time.time()
#                 # Check for double click
#                 if current_time - self.last_pinch_time < self.double_click_threshold:
#                     return "Double Click"
#                 self.last_pinch_time = current_time
#                 return "Left Click"
            
#             return "Pinch"
#         else:
#             self.prev_pinch_state = self.pinch_state
#             self.pinch_state = False
        
#         # Mid Tap (Right Click) - Thumb and Middle finger close
#         if thumb_middle_distance and thumb_middle_distance < self.pinch_threshold:
#             return "Right Click"
        
#         # Two-Finger Swipe (Scroll) - Index and Middle extended
#         if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             return "Two-Finger Swipe"
        
#         # App and Window Management Gestures
        
#         # Open Palm (Open App) - All fingers extended
#         if is_thumb_extended and is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
#             return "Open Palm"
        
#         # Fist (Close App) - No fingers extended
#         if not is_thumb_extended and not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             return "Fist"
        
#         # Peace Sign (Switch Tab) - Index and Middle extended in V shape
#         if is_index_extended and is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             # Check if fingers are spread in V shape
#             index_tip = self.landmarks.landmark[8]
#             middle_tip = self.landmarks.landmark[12]
#             distance = math.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
#             if distance > 0.1:  # Threshold for V shape
#                 return "Peace Sign"
        
#         # Circle (Change Window) - Thumb and Index form circle
#         thumb_tip = self.landmarks.landmark[4]
#         index_tip = self.landmarks.landmark[8]
#         if math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) < 0.05:
#             return "Circle"
        
#         # Five Finger Spread (Open Task View) - All fingers extended and spread
#         if is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
#             # Check if fingers are spread
#             index_tip = self.landmarks.landmark[8]
#             pinky_tip = self.landmarks.landmark[20]
#             distance = math.sqrt((index_tip.x - pinky_tip.x)**2 + (index_tip.y - pinky_tip.y)**2)
#             if distance > 0.2:  # Threshold for spread
#                 return "Five Finger Spread"
        
#         # Media Control Gestures
        
#         # Three-Finger Salute (Play/Pause) - Index, Middle, Ring extended
#         if is_index_extended and is_middle_extended and is_ring_extended and not is_pinky_extended:
#             return "Three-Finger Salute"
        
#         # Right/Left Swipe (Next/Previous Media) - Detected based on hand movement
#         if hand_center and self.prev_hand_center:
#             x_diff = hand_center[0] - self.prev_hand_center[0]
#             if abs(x_diff) > self.hand_movement_threshold:
#                 if x_diff > 0:
#                     return "Right Swipe"
#                 else:
#                     return "Left Swipe"
        
#         # Volume and Brightness Control
        
#         # Thumbs Up/Down (Volume Up/Down)
#         if is_thumb_extended and not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             thumb_tip = self.landmarks.landmark[4]
#             wrist = self.landmarks.landmark[0]
#             if thumb_tip.y < wrist.y:
#                 return "Thumbs Up"
#             else:
#                 return "Thumbs Down"
        
#         # L Shape (Screenshot) - Thumb and Index form L
#         if is_thumb_extended and is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
#             thumb_tip = self.landmarks.landmark[4]
#             index_tip = self.landmarks.landmark[8]
#             index_pip = self.landmarks.landmark[6]
            
#             # Check if thumb is horizontal and index is vertical
#             if abs(thumb_tip.y - index_pip.y) < 0.05 and abs(index_tip.x - index_pip.x) < 0.05:
#                 return "L Shape"
        
#         # Update previous hand center
#         self.prev_hand_center = hand_center
        
#         return "Unknown"
    
#     def get_smoothed_gesture(self):
#         if not self.gesture_history:
#             return "No Hand"
        
#         # Count occurrences of each gesture in history
#         gesture_counts = {}
#         for gesture in self.gesture_history:
#             if gesture in gesture_counts:
#                 gesture_counts[gesture] += 1
#             else:
#                 gesture_counts[gesture] = 1
        
#         # Return the most common gesture
#         return max(gesture_counts, key=gesture_counts.get)
    
#     def change_mode(self, direction=1):
#         current_time = time.time()
#         if current_time - self.mode_change_cooldown < 1:
#             return
        
#         self.current_mode = (self.current_mode + direction) % len(self.modes)
#         self.mode_change_cooldown = current_time
#         print(f"Switched to mode: {self.modes[self.current_mode]}")


# class HandController:
#     def __init__(self, recognizer):
#         self.recognizer = recognizer
#         self.mouse_smoothing = 0.5
#         self.prev_mouse_pos = pyautogui.position()
#         self.screen_width, self.screen_height = pyautogui.size()
        
#         # Action cooldowns
#         self.last_action_time = {}
#         self.cooldown_period = 0.5
        
#         # Scroll state
#         self.prev_scroll_pos = None
#         self.scroll_sensitivity = 30
        
#         # Volume control state
#         self.prev_volume_pos = None
#         self.volume_sensitivity = 5  # Adjusted for macOS
        
#         # Brightness control state
#         self.prev_brightness_pos = None
#         self.brightness_sensitivity = 10
        
#         # Media control state
#         self.media_cooldown = 0
        
#     def can_perform_action(self, action_name):
#         current_time = time.time()
#         if action_name in self.last_action_time:
#             if current_time - self.last_action_time[action_name] < self.cooldown_period:
#                 return False
        
#         self.last_action_time[action_name] = current_time
#         return True
    
#     def handle_gesture(self, frame):
#         if not self.recognizer.hand_present:
#             return frame
        
#         gesture = self.recognizer.get_smoothed_gesture()
#         mode = self.recognizer.modes[self.recognizer.current_mode]
        
#         # Display current mode and gesture
#         cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         # Mode switching with fist + swipe
#         if gesture == "Fist" and self.recognizer.prev_hand_center and self.recognizer.get_hand_center():
#             x_diff = self.recognizer.get_hand_center()[0] - self.recognizer.prev_hand_center[0]
#             if abs(x_diff) > 0.1:
#                 direction = 1 if x_diff > 0 else -1
#                 self.recognizer.change_mode(direction)
#                 return frame
        
#         # Handle gestures based on current mode
#         if mode == "Basic Control":
#             self.handle_basic_control(gesture, frame)
#         elif mode == "App Management":
#             self.handle_app_management(gesture, frame)
#         elif mode == "Media Control":
#             self.handle_media_control(gesture, frame)
#         elif mode == "Volume & Brightness":
#             self.handle_volume_brightness(gesture, frame)
#         elif mode == "Advanced":
#             self.handle_advanced_control(gesture, frame)
        
#         return frame
    
#     def handle_basic_control(self, gesture, frame):
#         # Point (Move Mouse)
#         if gesture == "Point":
#             if self.recognizer.landmarks:
#                 index_tip = self.recognizer.landmarks.landmark[8]
                
#                 # Convert normalized coordinates to screen coordinates
#                 target_x = int(index_tip.x * self.screen_width)
#                 target_y = int(index_tip.y * self.screen_height)
                
#                 # Apply smoothing
#                 current_x, current_y = pyautogui.position()
#                 new_x = current_x + (target_x - current_x) * self.mouse_smoothing
#                 new_y = current_y + (target_y - current_y) * self.mouse_smoothing
                
#                 # Move mouse
#                 pyautogui.moveTo(new_x, new_y)
                
#                 # Display mouse position
#                 cv2.putText(frame, f"Mouse: ({int(new_x)}, {int(new_y)})", (10, 90), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         # Left Click
#         elif gesture == "Left Click" and self.can_perform_action("left_click"):
#             pyautogui.click()
#             cv2.putText(frame, "Action: Left Click", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Double Click
#         elif gesture == "Double Click" and self.can_perform_action("double_click"):
#             pyautogui.doubleClick()
#             cv2.putText(frame, "Action: Double Click", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Right Click
#         elif gesture == "Right Click" and self.can_perform_action("right_click"):
#             pyautogui.rightClick()
#             cv2.putText(frame, "Action: Right Click", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Two-Finger Swipe (Scroll)
#         elif gesture == "Two-Finger Swipe":
#             if self.recognizer.landmarks:
#                 # Get the middle finger position for scrolling
#                 middle_tip = self.recognizer.landmarks.landmark[12]
                
#                 if self.prev_scroll_pos is None:
#                     self.prev_scroll_pos = middle_tip.y
#                 else:
#                     # Calculate scroll amount
#                     scroll_amount = int((middle_tip.y - self.prev_scroll_pos) * self.scroll_sensitivity)
                    
#                     if abs(scroll_amount) > 0:
#                         pyautogui.scroll(-scroll_amount)
#                         cv2.putText(frame, f"Scrolling: {-scroll_amount}", (10, 120), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
#                     self.prev_scroll_pos = middle_tip.y
#         else:
#             self.prev_scroll_pos = None
    
#     def handle_app_management(self, gesture, frame):
#         # Open Palm (Open App)
#         if gesture == "Open Palm" and self.can_perform_action("open_app"):
#             # macOS: Command+Space for Spotlight
#             pyautogui.hotkey('command', 'space')
#             cv2.putText(frame, "Action: Open Spotlight", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Fist (Close App)
#         elif gesture == "Fist" and self.can_perform_action("close_app"):
#             # macOS: Command+Q to quit current app
#             pyautogui.hotkey('command', 'q')
#             cv2.putText(frame, "Action: Quit App (Cmd+Q)", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Peace Sign (Switch Tab)
#         elif gesture == "Peace Sign" and self.can_perform_action("switch_tab"):
#             # macOS: Command+Tab to switch between apps
#             pyautogui.hotkey('command', 'tab')
#             cv2.putText(frame, "Action: Switch App (Cmd+Tab)", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Circle (Change Window)
#         elif gesture == "Circle" and self.can_perform_action("change_window"):
#             # macOS: Control+Down for Mission Control
#             pyautogui.hotkey('control', 'down')
#             cv2.putText(frame, "Action: Mission Control", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Five Finger Spread (Open Task View)
#         elif gesture == "Five Finger Spread" and self.can_perform_action("task_view"):
#             # macOS: F3 or Control+Up for Mission Control
#             pyautogui.hotkey('control', 'up')
#             cv2.putText(frame, "Action: Mission Control", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     def handle_media_control(self, gesture, frame):
#         # Three-Finger Salute (Play/Pause)
#         if gesture == "Three-Finger Salute" and self.can_perform_action("play_pause"):
#             # macOS: Play/Pause media key
#             pyautogui.press('playpause')
#             cv2.putText(frame, "Action: Play/Pause", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Right Swipe (Next Media)
#         elif gesture == "Right Swipe" and self.can_perform_action("next_media"):
#             # macOS: Next Track media key
#             pyautogui.press('nexttrack')
#             cv2.putText(frame, "Action: Next Track", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Left Swipe (Previous Media)
#         elif gesture == "Left Swipe" and self.can_perform_action("prev_media"):
#             # macOS: Previous Track media key
#             pyautogui.press('prevtrack')
#             cv2.putText(frame, "Action: Previous Track", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     def handle_volume_brightness(self, gesture, frame):
#         # Thumbs Up (Volume Up)
#         if gesture == "Thumbs Up" and self.can_perform_action("volume_up"):
#             # macOS: Volume up using osascript
#             subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) + 5)'])
#             cv2.putText(frame, "Action: Volume Up", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Thumbs Down (Volume Down)
#         elif gesture == "Thumbs Down" and self.can_perform_action("volume_down"):
#             # macOS: Volume down using osascript
#             subprocess.run(['osascript', '-e', 'set volume output volume (output volume of (get volume settings) - 5)'])
#             cv2.putText(frame, "Action: Volume Down", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Pinch (Volume/Brightness Control)
#         elif gesture == "Pinch":
#             if self.recognizer.landmarks:
#                 thumb_tip = self.recognizer.landmarks.landmark[4]
#                 index_tip = self.recognizer.landmarks.landmark[8]
                
#                 # Calculate pinch distance
#                 pinch_distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                
#                 # Volume control (vertical position)
#                 if thumb_tip.y < 0.3:  # Upper part of the frame
#                     # Map pinch distance to volume (0-100)
#                     volume_level = int(np.interp(pinch_distance, [0.02, 0.2], [0, 100]))
                    
#                     # Set volume using osascript
#                     subprocess.run(['osascript', '-e', f'set volume output volume {volume_level}'])
                    
#                     # Display volume level
#                     cv2.putText(frame, f"Volume: {volume_level}%", (10, 120), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
#                 # Brightness control (horizontal position)
#                 elif thumb_tip.x > 0.7:  # Right part of the frame
#                     try:
#                         # Map pinch distance to brightness
#                         brightness_level = int(np.interp(pinch_distance, [0.02, 0.2], [0, 100]))
#                         sbc.set_brightness(brightness_level)
                        
#                         # Display brightness level
#                         cv2.putText(frame, f"Brightness: {brightness_level}%", (10, 120), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     except Exception as e:
#                         # Fallback if screen_brightness_control doesn't work on this Mac
#                         cv2.putText(frame, "Brightness control not available", (10, 120), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     def handle_advanced_control(self, gesture, frame):
#         # L Shape (Screenshot)
#         if gesture == "L Shape" and self.can_perform_action("screenshot"):
#             # macOS: Command+Shift+3 for full screenshot
#             pyautogui.hotkey('command', 'shift', '3')
#             cv2.putText(frame, "Action: Screenshot", (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # Circle gesture for Undo/Redo
#         elif gesture == "Circle":
#             if self.recognizer.landmarks and self.recognizer.prev_hand_center and self.recognizer.get_hand_center():
#                 # Detect clockwise or counterclockwise movement
#                 prev_x, prev_y = self.recognizer.prev_hand_center
#                 curr_x, curr_y = self.recognizer.get_hand_center()
                
#                 # Simple detection of rotation direction
#                 if prev_x < curr_x and prev_y > curr_y:
#                     # Clockwise (Redo)
#                     if self.can_perform_action("redo"):
#                         pyautogui.hotkey('command', 'shift', 'z')  # macOS redo
#                         cv2.putText(frame, "Action: Redo (Cmd+Shift+Z)", (10, 120), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 elif prev_x > curr_x and prev_y > curr_y:
#                     # Counterclockwise (Undo)
#                     if self.can_perform_action("undo"):
#                         pyautogui.hotkey('command', 'z')  # macOS undo
#                         cv2.putText(frame, "Action: Undo (Cmd+Z)", (10, 120), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# def main():
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
    
#     # Set resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
#     # Initialize hand gesture recognizer
#     recognizer = HandGestureRecognizer()
    
#     # Initialize hand controller
#     controller = HandController(recognizer)
    
#     # Create window
#     cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_NORMAL)
    
#     # Instructions
#     print("=== Hand Gesture Control System for macOS ===")
#     print("Modes:")
#     for i, mode in enumerate(recognizer.modes):
#         print(f"{i+1}. {mode}")
#     print("\nUse Fist + Swipe Left/Right to change modes")
#     print("Press 'q' to quit")
    
#     while cap.isOpened():
#         # Read frame from webcam
#         success, frame = cap.read()
#         if not success:
#             print("Failed to read from webcam")
#             break
        
#         # Flip the frame horizontally for a more intuitive mirror view
#         frame = cv2.flip(frame, 1)
        
#         # Process the frame with MediaPipe Hands
#         frame = recognizer.process_frame(frame)
        
#         # Handle gestures and perform actions
#         frame = controller.handle_gesture(frame)
        
#         # Display the frame
#         cv2.imshow('Hand Gesture Control', frame)
        
#         # Exit on 'q' key press
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
    
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()




# project: Virtual Mouse
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Get screen size
screen_w, screen_h = pyautogui.size()
 
# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]  # Index finger tip
            middle_tip = landmarks[12]  # Middle finger tip
            thumb_tip = landmarks[4]  # Thumb tip
            thumb_knuckle = landmarks[5]  # Thumb base
            thumb_joint1 = landmarks[6]  # Thumb joint 1
            thumb_joint2 = landmarks[7]  # Thumb joint 2
            ring_tip = landmarks[16]  # Ring finger tip
            pinky_tip = landmarks[20]  # Pinky finger tip

            # Convert to screen coordinates
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            screen_x = np.interp(index_x, (0, w), (0, screen_w))
            screen_y = np.interp(index_y, (0, h), (0, screen_h))

            # Move Mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate distances for gesture recognition
            dist_index_middle = np.linalg.norm(
                np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y])
            )
            dist_thumb_5 = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([thumb_knuckle.x, thumb_knuckle.y])
            )
            dist_thumb_6 = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([thumb_joint1.x, thumb_joint1.y])
            )
            dist_thumb_7 = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([thumb_joint2.x, thumb_joint2.y])
            )
            dist_all_closed = np.linalg.norm(
                np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            ) + np.linalg.norm(
                np.array([middle_tip.x, middle_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            ) + np.linalg.norm(
                np.array([ring_tip.x, ring_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            ) + np.linalg.norm(
                np.array([pinky_tip.x, pinky_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )
            dist_control_tab = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
            ) + np.linalg.norm(
                np.array([middle_tip.x, middle_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
            )

            # Left Click (Thumb Touches Point 5, 6, or 7)
            if dist_thumb_5 < 0.05 or dist_thumb_6 < 0.05 or dist_thumb_7 < 0.05:
                pyautogui.click()

            # Right Click (Index + Middle Finger Touching)
            if dist_index_middle < 0.05:
                pyautogui.click(button='right')

            # Minimize Window (All Fingers Closed)
            if dist_all_closed < 0.15:
                pyautogui.hotkey('command', 'm')  # Minimize (Mac)
                # pyautogui.hotkey('alt', 'space', 'n')  # Minimize (Windows)

            # Control Floating Tab (Thumb + Index + Middle Holding)
            if dist_control_tab < 0.1:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
