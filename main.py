# Real-Time Object Detection & Logging Platform

import time

import streamlit as st
from streamlit_webrtc import webrtc_streamer

from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_CONFIDENCE,
    ICE_SERVERS,
    INFERENCE_IMAGE_SIZE,
    INFERENCE_MAX_SIDE,
    MODEL_PATH,
    PAGE_LAYOUT,
    RECENT_LOG_LIMIT,
)
from database import DatabaseManager, DetectionLogWorker
from detector import ObjectDetector
from ui import (
    create_video_area,
    show_header,
    show_recent_logs,
    show_sidebar,
)
from video_processor import LiveDetectionProcessor


@st.cache_resource
def load_detector():
    return ObjectDetector(
        model_path=MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
        image_size=INFERENCE_IMAGE_SIZE,
    )


@st.cache_data(ttl=10)
def load_recent_logs(limit):
    database = DatabaseManager()
    try:
        return database.get_recent_logs(limit)
    finally:
        database.close()


@st.cache_resource
def load_detection_log_worker():
    return DetectionLogWorker()


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
)

show_header()
confidence_threshold, object_option, start_detection = show_sidebar()
create_video_area()

if not start_detection:
    st.info("Camera stopped. Select **Start Camera** in the sidebar to begin.")
else:
    try:
        detector = load_detector()
    except Exception as error:
        st.error(f"Unable to load the YOLO model: {error}")
    else:
        log_worker = load_detection_log_worker()
        video_context = webrtc_streamer(
            key="live-camera-yolo",
            video_processor_factory=lambda: LiveDetectionProcessor(
                detector,
                log_worker,
                confidence_threshold,
                object_option,
            ),
            rtc_configuration={"iceServers": ICE_SERVERS},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": INFERENCE_MAX_SIDE},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30, "max": 30},
                },
                "audio": False,
            },
            desired_playing_state=True,
            media_toggle_controls=False,
            async_processing=False,
        )

        processor = video_context.video_processor
        if isinstance(processor, LiveDetectionProcessor):
            processor.update_settings(confidence_threshold, object_option)

        if video_context.state.playing:
            if (
                processor is not None
                and processor.last_frame_at is not None
                and time.monotonic() - processor.last_frame_at < 2.0
            ):
                st.success(
                    f"Live browser camera connected · Filter: **{object_option}** · "
                    f"Minimum confidence: **{confidence_threshold:.0%}**"
                )
            else:
                st.info(
                    "Camera connection is starting. Allow camera access in your "
                    "browser and wait for the first live frame."
                )
        else:
            st.info(
                "Waiting for browser camera permission and a WebRTC connection. "
                "The server cannot access your laptop's webcam directly."
            )

if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []
if "recent_logs_loaded" not in st.session_state:
    st.session_state.recent_logs_loaded = False
if "recent_logs_error" not in st.session_state:
    st.session_state.recent_logs_error = None

with st.expander("MySQL detection history"):
    refresh_history = st.button("Refresh detection history")
    if not st.session_state.recent_logs_loaded or refresh_history:
        try:
            if refresh_history:
                load_recent_logs.clear()
            st.session_state.recent_logs = load_recent_logs(RECENT_LOG_LIMIT)
            st.session_state.recent_logs_error = None
        except Exception as error:
            st.session_state.recent_logs_error = str(error)
        st.session_state.recent_logs_loaded = True

    if st.session_state.recent_logs_error:
        st.warning(
            "MySQL history is unavailable. Live video continues without "
            f"database access. Details: {st.session_state.recent_logs_error}"
        )
    show_recent_logs(st.session_state.recent_logs)
