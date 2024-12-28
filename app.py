import os
import sys

# Install required packages if they're not already installed
try:
    import cv2
except ImportError:
    os.system('pip install opencv-python-headless==4.8.1.78')
    import cv2

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from typing import Optional
import mediapipe as mp
import joblib

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load model and mappings (update the path as needed)
try:
    model = joblib.load(r"C:\Users\sanuv\OneDrive\Desktop\my_yoga_app\my_yoga_app\model\MediaPipe_Model.pkl")
except:
    st.error("Error loading model. Please check the model path.")


label_to_pose_name = {
 0: 'adho mukha svanasana',
 1: 'adho mukha vriksasana',
 2: 'agnistambhasana', 3: 'ananda balasana', 4: 'anantasana', 5: 'anjaneyasana', 6: 'ardha bhekasana', 7: 'ardha chandrasana', 8: 'ardha matsyendrasana', 9: 'ardha pincha mayurasana', 10: 'ardha uttanasana', 11: 'ashtanga namaskara', 12: 'astavakrasana', 13: 'baddha konasana', 14: 'bakasana', 15: 'balasana', 16: 'bhairavasana', 17: 'bharadvajasana i', 18: 'bhekasana', 19: 'bhujangasana', 20: 'bhujapidasana', 21: 'bitilasana', 22: 'camatkarasana', 23: 'chakravakasana', 24: 'chaturanga dandasana', 25: 'dandasana', 26: 'dhanurasana', 27: 'durvasasana', 28: 'dwi pada viparita dandasana', 29: 'eka pada koundinyanasana i', 30: 'eka pada koundinyanasana ii', 31: 'eka pada rajakapotasana', 32: 'eka pada rajakapotasana ii', 33: 'ganda bherundasana', 34: 'garbha pindasana', 35: 'garudasana', 36: 'gomukhasana', 37: 'halasana', 38: 'hanumanasana', 39: 'janu sirsasana', 40: 'kapotasana', 41: 'krounchasana', 42: 'kurmasana', 43: 'lolasana', 44: 'makara adho mukha svanasana', 45: 'makarasana', 46: 'malasana', 47: 'marichyasana i', 48: 'marichyasana iii', 49: 'marjaryasana', 50: 'matsyasana', 51: 'mayurasana', 52: 'natarajasana', 53: 'padangusthasana', 54: 'padmasana', 55: 'parighasana', 56: 'paripurna navasana', 57: 'parivrtta janu sirsasana', 58: 'parivrtta parsvakonasana', 59: 'parivrtta trikonasana', 60: 'parsva bakasana', 61: 'parsvottanasana', 62: 'pasasana', 63: 'paschimottanasana', 64: 'phalakasana', 65: 'pincha mayurasana', 66: 'prasarita padottanasana', 67: 'purvottanasana', 68: 'salabhasana', 69: 'salamba bhujangasana', 70: 'salamba sarvangasana', 71: 'salamba sirsasana', 72: 'savasana', 73: 'setu bandha sarvangasana', 74: 'simhasana', 75: 'sukhasana', 76: 'supta baddha konasana', 77: 'supta matsyendrasana', 78: 'supta padangusthasana', 79: 'supta virasana', 80: 'tadasana', 81: 'tittibhasana', 82: 'tolasana', 83: 'tulasana', 84: 'upavistha konasana', 85: 'urdhva dhanurasana', 86: 'urdhva hastasana', 87: 'urdhva mukha svanasana', 88: 'urdhva prasarita eka padasana', 89: 'ustrasana', 90: 'utkatasana', 91: 'uttana shishosana', 92: 'uttanasana', 93: 'utthita ashwa sanchalanasana', 94: 'utthita hasta padangustasana', 95: 'utthita parsvakonasana', 96: 'utthita trikonasana', 97: 'vajrasana', 98: 'vasisthasana', 99: 'viparita karani', 100: 'virabhadrasana i', 101: 'virabhadrasana ii', 102: 'virabhadrasana iii', 103: 'virasana', 104: 'vriksasana', 105: 'vrischikasana', 106: 'yoganidrasana'
}


# Reference angles for feedback
reference_angles = {
    "adho mukha svanasana": {
        "left_elbow": 150,
        "right_elbow": 150,
        "left_hip": 120,
        "right_hip": 120,
        "neck": 90,
        "shoulders": 120,
        # Add more joints as necessary
    },
    "adho mukha vriksasana": {
        "left_elbow": 180,
        "right_elbow": 180,
        "neck": 90,
        "shoulders": 120,
        # Add more joints
    },
    "agnistambhasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 90,
        "right_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ananda balasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 90,
        "right_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "anantasana": {
        "left_leg": 90,  # Leg extended upwards
        "right_leg": 0,  # Bent leg support
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "anjaneyasana": {
        "left_knee": 90,
        "right_knee": 180,
        "left_hip": 90,
        "right_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ardha bhekasana": {
        "left_leg": 90,
        "right_leg": 0,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ardha chandrasana": {
        "left_leg": 90,
        "right_leg": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ardha matsyendrasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 60,
        "right_hip": 60,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ardha pincha mayurasana": {
        "left_elbow": 150,
        "right_elbow": 150,
        "neck": 90,
        "shoulders": 120,
        # Add more joints
    },
    "ardha uttanasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ashtanga namaskara": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "astavakrasana": {
        "left_leg": 90,
        "right_leg": 90,
        "left_hip": 45,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "baddha konasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bakasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "balasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bhairavasana": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 60,
        "right_hip": 60,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bharadvajasana i": {
        "left_knee": 90,
        "right_knee": 90,
        "left_hip": 60,
        "right_hip": 60,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bhekasana": {
        "left_leg": 90,
        "right_leg": 0,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bhujangasana": {
        "left_elbow": 120,
        "right_elbow": 120,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bhujapidasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "bitilasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "camatkarasana": {
        "left_leg": 90,
        "right_leg": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "chakravakasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "chaturanga dandasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "dandasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "dhanurasana": {
        "left_leg": 90,
        "right_leg": 90,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "durvasasana": {
        "left_leg": 90,
        "right_leg": 90,
        "left_hip": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "dwi pada viparita dandasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "eka pada koundinyanasana i": {
        "left_leg": 90,
        "right_leg": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "eka pada koundinyanasana ii": {
        "left_leg": 90,
        "right_leg": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "eka pada rajakapotasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "eka pada rajakapotasana ii": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "ganda bherundasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "garbha pindasana": {
        "left_leg": 90,
        "right_leg": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "garudasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "gomukhasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "halasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "hanumanasana": {
        "left_leg": 90,
        "right_leg": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "janu sirsasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "kapotasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "krounchasana": {
        "left_leg": 90,
        "right_leg": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "kurmasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "lolasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "makara adho mukha svanasana": {
        "left_elbow": 150,
        "right_elbow": 150,
        "neck": 90,
        "shoulders": 120,
        # Add more joints
    },
    "makarasana": {
        "left_elbow": 120,
        "right_elbow": 120,
        "neck": 0,
        "hip": 180,
        "shoulders": 120,
        # Add more joints
    },
    "malasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "marichyasana i": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "marichyasana iii": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "marjaryasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "matsyasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "mayurasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "natarajasana": {
        "left_leg": 90,
        "right_leg": 0,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "padangusthasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "padmasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        # Add more joints
    },
    "parighasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "paripurna navasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "parivrtta janu sirsasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "parivrtta parsvakonasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "parivrtta trikonasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "parsva bakasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "parsvottanasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "pasasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "paschimottanasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "phalakasana": {
        "left_elbow": 90,
        "right_elbow": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "pincha mayurasana": {
        "left_elbow": 150,
        "right_elbow": 150,
        "neck": 90,
        "shoulders": 120,
        # Add more joints
    },
    "prasarita padottanasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "purvottanasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "salabhasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "salamba bhujangasana": {
        "left_elbow": 120,
        "right_elbow": 120,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "salamba sarvangasana": {
        "left_knee": 180,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "salamba sirsasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "savasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "setu bandha sarvangasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "simhasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "sukhasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "supta baddha konasana": {
        "left_knee": 90,
        "right_knee": 90,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "supta matsyendrasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    },
    "supta matsyasana": {
        "left_knee": 90,
        "right_knee": 180,
        "neck": 0,
        "shoulders": 120,
        # Add more joints
    }
  }



def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)



def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_angles(landmarks):
    """
    Extracts angles between key body points using Mediapipe landmarks.

    Parameters:
        landmarks: A list of Mediapipe landmarks.

    Returns:
        A dictionary of angles with body parts as keys.
    """
    angles = {}

    # Calculate angles for left and right arms
    angles["left_elbow"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )

    angles["right_elbow"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )

    # Calculate angles for left and right knees
    angles["left_knee"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    )

    angles["right_knee"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    )

    # Calculate the neck angle
    angles["neck"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
         landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    )

    # Calculate hip and shoulder angles
    angles["left_hip"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    )

    angles["right_hip"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    )

    angles["left_shoulder"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
         landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    )

    angles["right_shoulder"] = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
         landmarks[mp_pose.PoseLandmark.NOSE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    )

    return angles

def get_reference_image_link(predicted_pose: str) -> Optional[str]:
    """Get the reference image link for the given predicted pose."""
    text_file_path = r"C:\Users\sanuv\OneDrive\Desktop\my_yoga_app\my_yoga_app\pose links.txt"
    
    try:
        with open(text_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            pose_name, image_link = parts[0].strip().lower(), parts[1].strip()

            if pose_name == predicted_pose.lower():
                return image_link

        return None
    except FileNotFoundError:
        st.error("Text file for pose links not found")
        return None
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def predict_and_visualize(image):
    """Predict the yoga pose and provide feedback."""
    try:
        # Process the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return "No pose detected", None, None, None

        landmarks = results.pose_landmarks.landmark
        keypoints = [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]
        predicted_label = model.predict([keypoints])[0]
        predicted_pose = label_to_pose_name.get(predicted_label, "Unknown Pose")

        # Generate feedback
        angles = extract_angles(landmarks)
        feedback = []
        if predicted_pose in reference_angles:
            for joint, ref_angle in reference_angles[predicted_pose].items():
                user_angle = angles.get(joint, None)
                if user_angle and abs(user_angle - ref_angle) > 10:
                    feedback.append(f"Adjust your {joint}, off by {abs(user_angle - ref_angle):.2f}°")

        feedback_message = feedback if feedback else ["Great job! Your pose matches the reference."]

        # Draw landmarks
        mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get reference image link
        reference_image_link = get_reference_image_link(predicted_pose)

        return predicted_pose, feedback_message, reference_image_link, image_rgb

    except Exception as e:
        return str(e), None, None, None

def main():
    st.title("Yoga Pose Detection")
    st.write("Upload an image to detect and analyze your yoga pose!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Image")
            st.image(image, channels="BGR", use_column_width=True)

        # Process the image and get results
        predicted_pose, feedback, reference_image_link, visualization = predict_and_visualize(image)

        with col2:
            st.subheader("Pose Detection")
            if visualization is not None:
                st.image(visualization, channels="RGB", use_column_width=True)

        # Display results
        st.subheader("Results")
        st.write(f"**Predicted Pose:** {predicted_pose}")

        # Display feedback
        st.subheader("Feedback")
        if feedback:
            for item in feedback:
                st.write(f"- {item}")

        # Display reference image if available
        if reference_image_link:
            st.subheader("Reference Image")
            st.image(reference_image_link)

if __name__ == "__main__":
    main()