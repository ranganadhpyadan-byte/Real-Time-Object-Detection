# config.py
# Configuration Module
# Assignment 5 - Real-Time Object Detection & Logging Platform

import os
from dotenv import load_dotenv


# ---------------------------------------------------------
# LOAD .ENV FILE
# ---------------------------------------------------------
load_dotenv()


# =========================================================
# APPLICATION SETTINGS
# =========================================================

APP_TITLE = "Real-Time Object Detection & Logging Platform"

APP_ICON = "🤖"


# =========================================================
# YOLO SETTINGS
# =========================================================

# YOLO model
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "yolov8n.pt"
)

# Default confidence threshold
DEFAULT_CONFIDENCE = float(
    os.getenv(
        "DEFAULT_CONFIDENCE",
        "0.70"
    )
)

# Minimum confidence allowed in UI
MIN_CONFIDENCE = 0.10

# Maximum confidence allowed in UI
MAX_CONFIDENCE = 1.00

# Confidence slider step
CONFIDENCE_STEP = 0.05


# =========================================================
# CAMERA SETTINGS
# =========================================================

# Default webcam index
CAMERA_INDEX = int(
    os.getenv(
        "CAMERA_INDEX",
        "0"
    )
)

# Camera frame width
FRAME_WIDTH = int(
    os.getenv(
        "FRAME_WIDTH",
        "640"
    )
)

# Camera frame height
FRAME_HEIGHT = int(
    os.getenv(
        "FRAME_HEIGHT",
        "480"
    )
)


# =========================================================
# OBJECT DETECTION SETTINGS
# =========================================================

# Objects available in the Streamlit filter
OBJECT_CLASSES = [
    "All Objects",
    "person",
    "cell phone",
    "laptop",
    "bottle",
    "chair",
    "car",
    "dog",
    "cat"
]


# =========================================================
# DATABASE SETTINGS
# =========================================================

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost"
)

DB_USER = os.getenv(
    "DB_USER",
    "root"
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    ""
)

DB_NAME = os.getenv(
    "DB_NAME",
    "vision_platform"
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "3306"
    )
)


# =========================================================
# DATABASE TABLE
# =========================================================

DB_TABLE = "detection_logs"


# =========================================================
# LOGGING SETTINGS
# =========================================================

# Number of recent records shown in UI
RECENT_LOG_LIMIT = int(
    os.getenv(
        "RECENT_LOG_LIMIT",
        "20"
    )
)


# =========================================================
# STREAMLIT SETTINGS
# =========================================================

PAGE_LAYOUT = "wide"


# =========================================================
# DETECTION LOGGING COOLDOWN
# =========================================================

# Minimum time in seconds before logging
# the same object class again.
#
# This prevents the database from being filled
# with hundreds of identical records per second.

LOG_COOLDOWN = float(
    os.getenv(
        "LOG_COOLDOWN",
        "2.0"
    )
)