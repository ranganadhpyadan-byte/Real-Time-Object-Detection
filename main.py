# main.py
# Real-Time Object Detection & Logging Platform
# Assignment 5

import time

import cv2
import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LOG_COOLDOWN,
    MODEL_PATH,
    PAGE_LAYOUT,
    RECENT_LOG_LIMIT,
)
from database import DatabaseManager
from detector import ObjectDetector
from ui import (
    create_video_area,
    show_header,
    show_metrics,
    show_object_statistics,
    show_recent_logs,
    show_sidebar,
    show_welcome,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
show_header()


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
confidence_threshold, object_option, start_detection = show_sidebar()


# ---------------------------------------------------------
# INITIALIZE OBJECT DETECTOR
# ---------------------------------------------------------
@st.cache_resource
def load_detector():
    return ObjectDetector(
        model_path=MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
    )


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------
@st.cache_resource
def load_database():
    try:
        return DatabaseManager()
    except Exception as error:
        return error


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
if "camera" not in st.session_state:
    st.session_state.camera = None
if "detection_counts" not in st.session_state:
    st.session_state.detection_counts = {}
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "last_logged" not in st.session_state:
    st.session_state.last_logged = {}
if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []
if "last_log_refresh" not in st.session_state:
    st.session_state.last_log_refresh = 0.0

if start_detection and st.session_state.camera is None:
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not camera.isOpened():
            camera.release()
            st.session_state.camera_running = False
            st.error(
                f"Unable to open webcam {CAMERA_INDEX}. Check camera permissions "
                "or configure CAMERA_INDEX."
            )
        else:
            st.session_state.camera = camera
    except Exception as error:
        st.session_state.camera_running = False
        st.error(f"Unable to start webcam: {error}")
elif not start_detection and st.session_state.camera is not None:
    st.session_state.camera.release()
    st.session_state.camera = None


# Read and infer one frame per fragment tick instead of blocking Streamlit in a
# while-loop, so Stop and filter changes remain responsive during live streaming.
@st.fragment(run_every="50ms")
def render_live_detection():
    if not start_detection or st.session_state.camera is None:
        show_welcome(confidence_threshold, object_option)
        return

    camera = st.session_state.camera
    if not camera.isOpened():
        camera.release()
        st.session_state.camera = None
        st.session_state.camera_running = False
        st.error("The webcam is no longer available. The camera was released.")
        return

    st.caption(
        f"🟢 Camera active · Filter: **{object_option}** · "
        f"Minimum confidence: **{confidence_threshold:.0%}**"
    )
    success, frame = camera.read()
    if not success:
        camera.release()
        st.session_state.camera = None
        st.session_state.camera_running = False
        st.error("Could not read a webcam frame. The camera was stopped and released.")
        return

    try:
        detector = load_detector()
    except Exception as error:
        camera.release()
        st.session_state.camera = None
        st.session_state.camera_running = False
        st.error(f"Unable to load the YOLO model: {error}")
        return

    detector.set_confidence(confidence_threshold)
    try:
        detections = detector.detect(frame)
    except Exception as error:
        st.error(f"YOLO inference failed: {error}")
        return

    accepted_detections = []
    database = load_database()
    now = time.monotonic()
    database_error = None

    for detection in detections:
        object_class = detection["class"]
        confidence = detection["confidence"]
        if object_option != "All Objects" and object_class != object_option:
            continue
        if confidence < confidence_threshold:
            continue

        accepted_detections.append((object_class, confidence))
        x, y = detection["x"], detection["y"]
        width, height = detection["w"], detection["h"]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{object_class}: {confidence * 100:.1f}%",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        st.session_state.total_detections += 1
        counts = st.session_state.detection_counts
        counts[object_class] = counts.get(object_class, 0) + 1

        last_logged = st.session_state.last_logged.get(object_class, 0.0)
        if now - last_logged < LOG_COOLDOWN:
            continue

        if isinstance(database, Exception):
            database_error = str(database)
            continue

        st.session_state.last_logged[object_class] = now
        try:
            database.insert_detection(
                object_class=object_class,
                confidence=confidence,
                bbox_x=x,
                bbox_y=y,
                bbox_w=width,
                bbox_h=height,
            )
        except Exception as error:
            database_error = str(error)

    if object_option in ("All Objects", "person"):
        person_visible = any(
            detection["class"] == "person"
            and detection["confidence"] >= confidence_threshold
            for detection in detections
        )
        stream_status = (
            "PERSON DETECTED"
            if person_visible
            else f"NO PERSON DETECTED · LIVE VIDEO CONTINUES · {confidence_threshold:.0%} THRESHOLD"
        )
    else:
        stream_status = (
            f"{len(accepted_detections)} MATCHING DETECTION(S)"
            if accepted_detections
            else f"NO {object_option.upper()} DETECTED · LIVE VIDEO CONTINUES"
        )

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (15, 23, 42), -1)
    cv2.putText(
        frame,
        stream_status,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    most_detected = max(
        st.session_state.detection_counts,
        key=st.session_state.detection_counts.get,
        default="None",
    )
    show_metrics(
        st.session_state.total_detections,
        most_detected,
        confidence_threshold,
    )

    col_video, col_statistics = st.columns([2, 1])
    with col_video:
        video_placeholder = create_video_area()
        video_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True,
        )
    with col_statistics:
        show_object_statistics(st.session_state.detection_counts)

    if accepted_detections:
        visible_detections = ", ".join(
            f"{object_class} ({confidence:.0%})"
            for object_class, confidence in accepted_detections[:5]
        )
        if len(accepted_detections) > 5:
            visible_detections += f", +{len(accepted_detections) - 5} more"
        st.success(f"Detected this frame: {visible_detections}", icon="✅")
    elif object_option == "person" or object_option == "All Objects":
        st.info(
            f"No person detected above {confidence_threshold:.0%} confidence in this frame. "
            "The live camera is still running; try better lighting, move closer, "
            "or lower the threshold."
        )
    else:
        st.info(
            f"No **{object_option}** detected above {confidence_threshold:.0%} confidence. "
            "The live camera is still running."
        )

    if database_error:
        st.warning(
            "MySQL logging is unavailable; detection will continue without saving events. "
            f"Details: {database_error}"
        )
    elif (
        isinstance(database, DatabaseManager)
        and database.connection is not None
        and database.connection.is_connected()
    ):
        if now - st.session_state.last_log_refresh >= 2.0:
            try:
                st.session_state.recent_logs = database.get_recent_logs(
                    RECENT_LOG_LIMIT
                )
                st.session_state.last_log_refresh = now
            except Exception as error:
                database_error = str(error)
        show_recent_logs(st.session_state.recent_logs)
        if database_error:
            st.warning(f"Unable to refresh recent MySQL logs: {database_error}")
    elif isinstance(database, DatabaseManager):
        st.warning(
            "MySQL is not connected; detection is running without database logging. "
            f"Details: {database.last_error or 'Connection is unavailable.'}"
        )
    elif isinstance(database, Exception):
        st.warning(
            "MySQL is not connected; detection is running without database logging. "
            f"Details: {database}"
        )
    else:
        st.info("MySQL is not connected. Detection is running without database logging.")

render_live_detection()