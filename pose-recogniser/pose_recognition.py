#imports 
import cv2
import mediapipe as mp
import numpy as np
import math

#Intialise Mediapipe
mp_hollistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

#Start Camera Capture
cap = cv2.VideoCapture(0)

try:
    img_sybau = cv2.imread('sybau.png',cv2.IMREAD_UNCHANGED)
    img_angry = cv2.imread('grr.png',cv2.IMREAD_UNCHANGED)
    img_smirk = cv2.imread('bofem.png',cv2.IMREAD_UNCHANGED)
    img_happy = cv2.imread('happy.png',cv2.IMREAD_UNCHANGED)
    img_freaky = cv2.imread('freaky.png',cv2.IMREAD_UNCHANGED)
except Exception as e:
    print("Error: Could not load one or more image")
    print("Make sure that all photos are in the same folder")
    print(e) #Prints actual error
    #Exits the program
    cap.release()
    exit()
    
#Main Code/Logic
with mp_hollistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #Pose Window
    cv2.namedWindow('Pose ahh', cv2.WINDOW_NORMAL)
    
    #Pose Window Size
    POSE_WINDOW_HEIGHT = 500
    POSE_WINDOW_WIDTH = 500