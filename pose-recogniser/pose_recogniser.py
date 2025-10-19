# Pose & Face Recognizer
# This script uses your camera to detect facial expressions and hand gestures
# and displays a corresponding PNG image in a separate window.

import cv2
import mediapipe as mp
import numpy as np
import math

# --- Initialization ---
# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load PNG images at their original size
try:
    img_sybau = cv2.imread('sybau.png', cv2.IMREAD_UNCHANGED)
    img_happy = cv2.imread('happy.png', cv2.IMREAD_UNCHANGED)
    img_grr = cv2.imread('grr.png', cv2.IMREAD_UNCHANGED)
    img_freaky = cv2.imread('freaky.png', cv2.IMREAD_UNCHANGED)
    img_bofem = cv2.imread('bofem.png', cv2.IMREAD_UNCHANGED) # New image

    # Add a check to ensure all images were loaded successfully
    if img_sybau is None or img_happy is None or img_grr is None or img_freaky is None or img_bofem is None:
        raise IOError("One or more images failed to load.")

except Exception as e:
    print("Error: Could not read one or more images.")
    print("Make sure 'sybau.png', 'happy.png', 'grr.png', 'freaky.png', and 'bofem.png' are in the same folder as the script.")
    print(e) # Print the actual error
    cap.release()
    exit()

# --- Main Loop ---
# Initialize Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Create the named window for the pose image so we can reference it
    cv2.namedWindow('Pose Image', cv2.WINDOW_NORMAL)

    # Define the size for the second window. You can change these values.
    POSE_WINDOW_WIDTH = 500
    POSE_WINDOW_HEIGHT = 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find landmarks
        results = holistic.process(image_rgb)
        
        # Convert the image color back so it can be displayed
        frame.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # --- Pose Detection Logic ---
        image_height, image_width, _ = frame.shape
        pose_status = ""
        debug_brow_dist = 0.0 # Variable to hold the value for debugging
        debug_smirk_val = 0.0 # New debug value for smug face

        # We need to access landmarks, so we'll check if they exist first
        if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            face_landmarks = results.face_landmarks.landmark
            
            # -- Logic for Finger over Mouth ("sybau.png") --
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    mouth_center_x = face_landmarks[13].x
                    mouth_center_y = face_landmarks[13].y
                    finger_tip_x = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x
                    finger_tip_y = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y
                    
                    distance = math.sqrt((finger_tip_x - mouth_center_x)**2 + (finger_tip_y - mouth_center_y)**2)
                    if distance < 0.1:
                        pose_status = "Shush"

        # Check for facial expressions only if a hand pose wasn't detected
        if pose_status == "" and results.face_landmarks:
            face_landmarks = results.face_landmarks.landmark

            # Normalize by face height to make it scale-independent
            face_top = face_landmarks[10]
            face_bottom = face_landmarks[152]
            face_height = abs(face_top.y - face_bottom.y)

            # Avoid division by zero if face height is not detected
            if face_height > 0:
                # -- Logic for Mouth Open ("happy.png") and Tongue Out ("freaky.png") --
                upper_lip = face_landmarks[13]
                lower_lip = face_landmarks[14]
                mouth_open_distance = abs(upper_lip.y - lower_lip.y)
                normalized_mouth_dist = mouth_open_distance / face_height

                nose_tip = face_landmarks[1]
                chin = face_landmarks[152]
                nose_chin_distance = abs(nose_tip.y - chin.y)
                normalized_nose_chin_dist = nose_chin_distance / face_height

                if normalized_mouth_dist > 0.14 and normalized_nose_chin_dist > 0.35:
                    pose_status = "Tongue Out"
                elif normalized_mouth_dist > 0.08:
                    pose_status = "Happy"
                else:
                    # -- Logic for Smug Face ("bofem.png") --
                    left_mouth_corner = face_landmarks[291]
                    right_mouth_corner = face_landmarks[61]
                    smirk_diff = abs(left_mouth_corner.y - right_mouth_corner.y)
                    normalized_smirk_diff = smirk_diff / face_height
                    debug_smirk_val = normalized_smirk_diff
                    
                    if normalized_smirk_diff > 0.02: # Threshold for a smirk
                        pose_status = "Smug"

            # -- Logic for Angry Face ("grr.png") --
            if pose_status == "":
                left_eyebrow = face_landmarks[282]
                right_eyebrow = face_landmarks[52]
                brow_distance = abs(left_eyebrow.x - right_eyebrow.x)

                face_left = face_landmarks[454]
                face_right = face_landmarks[234]
                face_width = abs(face_left.x - face_right.x)
                
                if face_width > 0:
                    normalized_brow_dist = brow_distance / face_width
                    debug_brow_dist = normalized_brow_dist
                    if normalized_brow_dist < 0.559:
                        pose_status = "Angry"

        # --- Prepare Image for Second Window ---
        pose_image_display = np.zeros((POSE_WINDOW_HEIGHT, POSE_WINDOW_WIDTH, 3), dtype=np.uint8)

        img_to_show = None
        if pose_status == "Shush":
            img_to_show = img_sybau
        elif pose_status == "Tongue Out":
            img_to_show = img_freaky
        elif pose_status == "Happy":
            img_to_show = img_happy
        elif pose_status == "Smug": # New case
            img_to_show = img_bofem
        elif pose_status == "Angry":
            img_to_show = img_grr

        if img_to_show is not None:
            img_color = img_to_show
            if img_to_show.shape[2] == 4:
                img_color = img_to_show[:, :, :3]
            
            TARGET_WIDTH = 400
            h, w, _ = img_color.shape
            aspect_ratio = w / h
            new_w = TARGET_WIDTH
            new_h = int(new_w / aspect_ratio)
            resized_img = cv2.resize(img_color, (new_w, new_h))

            y_offset = max((POSE_WINDOW_HEIGHT - new_h) // 2, 0)
            x_offset = max((POSE_WINDOW_WIDTH - new_w) // 2, 0)

            end_y = min(y_offset + new_h, POSE_WINDOW_HEIGHT)
            end_x = min(x_offset + new_w, POSE_WINDOW_WIDTH)
            
            pose_image_display[y_offset:end_y, x_offset:end_x] = resized_img[0:end_y-y_offset, 0:end_x-x_offset]


        # Display the status text on the main camera frame
        cv2.putText(frame, f"Status: {pose_status}", (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the debug value for brow distance
        cv2.putText(frame, f"Brow Dist: {round(debug_brow_dist, 3)}", (10, image_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the new debug value for smug/smirk
        cv2.putText(frame, f"Smirk Val: {round(debug_smirk_val, 3)}", (10, image_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # --- Show the windows ---
        cv2.imshow('Facial and Pose Recognition', frame)
        cv2.imshow('Pose Image', pose_image_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()

