# config.py
# Configuration Module
# Assignment 5 - Real-Time Object Detection & Logging Platform

import os

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


# ---------------------------------------------------------
# LOAD .ENV FILE
# ---------------------------------------------------------
load_dotenv()


def _setting(name, default):
    if name in os.environ:
        return os.environ[name]

    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


# =========================================================
# APPLICATION SETTINGS
# =========================================================

APP_TITLE = "Real-Time Object Detection & Logging Platform"

APP_ICON = "🤖"


# =========================================================
# YOLO SETTINGS
# =========================================================

# YOLO model
MODEL_PATH = _setting(
    "MODEL_PATH",
    "yolov8n.pt"
)

# Default confidence threshold
DEFAULT_CONFIDENCE = float(
    _setting(
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
    _setting(
        "CAMERA_INDEX",
        "0"
    )
)

# Camera frame width
FRAME_WIDTH = int(
    _setting(
        "FRAME_WIDTH",
        "640"
    )
)

# Camera frame height
FRAME_HEIGHT = int(
    _setting(
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

DB_HOST = _setting(
    "DB_HOST",
    "localhost"
)

DB_USER = _setting(
    "DB_USER",
    "root"
)

DB_PASSWORD = _setting(
    "DB_PASSWORD",
    ""
)

DB_NAME = _setting(
    "DB_NAME",
    "vision_platform"
)

DB_PORT = int(
    _setting(
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
    _setting(
        "RECENT_LOG_LIMIT",
        "20"
    )
)


# =========================================================
# STREAMLIT SETTINGS
# =========================================================

PAGE_LAYOUT = "wide"

ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
]
TURN_SERVER_URLS = _setting("TURN_SERVER_URLS", "").strip()
TURN_SERVER_USERNAME = _setting("TURN_SERVER_USERNAME", "").strip()
TURN_SERVER_CREDENTIAL = _setting("TURN_SERVER_CREDENTIAL", "").strip()
turn_server_urls = [url.strip() for url in TURN_SERVER_URLS.split(",") if url.strip()]
if any((TURN_SERVER_URLS, TURN_SERVER_USERNAME, TURN_SERVER_CREDENTIAL)):
    if not all(
        (turn_server_urls, TURN_SERVER_USERNAME, TURN_SERVER_CREDENTIAL)
    ):
        raise ValueError(
            "Configure TURN_SERVER_URLS, TURN_SERVER_USERNAME, and "
            "TURN_SERVER_CREDENTIAL together."
        )
    ICE_SERVERS.append(
        {
            "urls": turn_server_urls,
            "username": TURN_SERVER_USERNAME,
            "credential": TURN_SERVER_CREDENTIAL,
        }
    )


# =========================================================
# DETECTION LOGGING COOLDOWN
# =========================================================

# Minimum time in seconds before logging
# the same object class again.
#
# This prevents the database from being filled
# with hundreds of identical records per second.

LOG_COOLDOWN = float(
    _setting(
        "LOG_COOLDOWN",
        "2.0"
    )
)

# Keep camera frames flowing while limiting YOLO work on the server.
INFERENCE_FPS = float(_setting("INFERENCE_FPS", "7"))
INFERENCE_MAX_SIDE = int(_setting("INFERENCE_MAX_SIDE", "640"))
INFERENCE_IMAGE_SIZE = int(_setting("INFERENCE_IMAGE_SIZE", "416"))

if not 0 < INFERENCE_FPS <= 10:
    raise ValueError("INFERENCE_FPS must be greater than 0 and at most 10.")
if INFERENCE_MAX_SIDE < 1 or INFERENCE_IMAGE_SIZE < 1:
    raise ValueError("Inference frame dimensions must be positive integers.")
if not LOG_COOLDOWN >= 1:
    raise ValueError("LOG_COOLDOWN must be at least 1 second.")