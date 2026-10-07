# Real-Time Object Detection & Logging Platform

import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_CONFIDENCE,
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
    show_recent_logs,
    show_sidebar,
)


@st.cache_resource
def load_detector():
    return ObjectDetector(
        model_path=MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
    )


@st.cache_resource
def load_database():
    return DatabaseManager()


class LiveDetectionProcessor(VideoProcessorBase):
    def __init__(self, detector, confidence_threshold, object_filter):
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.object_filter = object_filter
        self.last_logged = {}
        self.database = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        detections = self.detector.detect(
            image,
            confidence=self.confidence_threshold,
        )
        accepted = []

        for detection in detections:
            object_class = detection["class"]
            confidence = detection["confidence"]
            if (
                self.object_filter != "All Objects"
                and object_class != self.object_filter
            ):
                continue
            if confidence < self.confidence_threshold:
                continue

            accepted.append(detection)
            x, y = detection["x"], detection["y"]
            width, height = detection["w"], detection["h"]
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"{object_class}: {confidence * 100:.1f}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            self._log_detection(detection)

        status = (
            f"LIVE VIDEO · {len(accepted)} DETECTION(S)"
            if accepted
            else "LIVE VIDEO · NO MATCHING OBJECTS"
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 38), (15, 23, 42), -1)
        cv2.putText(
            image,
            status,
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def _log_detection(self, detection):
        object_class = detection["class"]
        now = time.monotonic()
        if now - self.last_logged.get(object_class, 0.0) < LOG_COOLDOWN:
            return

        self.last_logged[object_class] = now
        if self.database is None:
            self.database = DatabaseManager()

        try:
            self.database.insert_detection(
                object_class=object_class,
                confidence=detection["confidence"],
                bbox_x=detection["x"],
                bbox_y=detection["y"],
                bbox_w=detection["w"],
                bbox_h=detection["h"],
            )
        except Exception as error:
            print(f"MySQL detection logging failed; video continues: {error}")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
)

show_header()
confidence_threshold, object_option, start_detection = show_sidebar()
create_video_area()

try:
    detector = load_detector()
except Exception as error:
    st.error(f"Unable to load the YOLO model: {error}")
else:
    if start_detection:
        st.caption(
            f"🟢 Camera active · Filter: **{object_option}** · "
            f"Minimum confidence: **{confidence_threshold:.0%}**"
        )
    else:
        st.info("Camera stopped. Select **Start Camera** in the sidebar to resume.")

    webrtc_streamer(
        key="live-camera-yolo",
        video_processor_factory=lambda: LiveDetectionProcessor(
            detector,
            confidence_threshold,
            object_option,
        ),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=start_detection,
        media_toggle_controls=False,
        async_processing=False,
    )

if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []

with st.expander("MySQL detection history"):
    if st.button("Refresh detection history"):
        try:
            st.session_state.recent_logs = load_database().get_recent_logs(
                RECENT_LOG_LIMIT
            )
        except Exception as error:
            st.warning(
                "MySQL history is unavailable. Detection video continues without "
                f"database access. Details: {error}"
            )
    show_recent_logs(st.session_state.recent_logs)
