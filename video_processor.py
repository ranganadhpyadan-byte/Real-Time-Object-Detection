import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from config import (
    INFERENCE_FPS,
    INFERENCE_MAX_SIDE,
    LOG_COOLDOWN,
)

logger = logging.getLogger(__name__)


class LiveDetectionProcessor(VideoProcessorBase):
    def __init__(
        self,
        detector,
        log_worker,
        confidence_threshold,
        object_filter,
    ):
        self.detector = detector
        self.log_worker = log_worker
        self.confidence_threshold = confidence_threshold
        self.object_filter = object_filter
        self.last_logged = {}
        self._settings_lock = threading.Lock()
        self._last_inference_at = 0.0
        self._last_error_log_at = 0.0
        self._detections = []
        self._detection_error = False
        self._settings_generation = 0
        self._inference_future = None
        self._inference_generation = None
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="yolo-inference",
        )
        self.last_frame_at = None

    def update_settings(self, confidence_threshold, object_filter):
        with self._settings_lock:
            changed = (
                self.confidence_threshold != confidence_threshold
                or self.object_filter != object_filter
            )
            self.confidence_threshold = confidence_threshold
            self.object_filter = object_filter
            if changed:
                self._settings_generation += 1
                self._detections = []
                self._last_inference_at = 0.0

    def on_ended(self):
        self._inference_executor.shutdown(wait=False, cancel_futures=True)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        self.last_frame_at = now
        with self._settings_lock:
            confidence_threshold = self.confidence_threshold
            object_filter = self.object_filter
            settings_generation = self._settings_generation

        fresh_detections = False
        if (
            self._inference_future is not None
            and self._inference_future.done()
        ):
            completed_future = self._inference_future
            completed_generation = self._inference_generation
            self._inference_future = None
            self._inference_generation = None
            if completed_generation == settings_generation:
                try:
                    detections = completed_future.result()
                    with self._settings_lock:
                        self._detections = detections
                        self._detection_error = False
                    fresh_detections = True
                except Exception:
                    with self._settings_lock:
                        self._detections = []
                        self._detection_error = True
                    if now - self._last_error_log_at >= 30:
                        logger.exception(
                            "YOLO inference failed; live video continues."
                        )
                        self._last_error_log_at = now

        if (
            self._inference_future is None
            and now - self._last_inference_at >= 1.0 / INFERENCE_FPS
        ):
            frame_height, frame_width = image.shape[:2]
            scale = min(
                1.0,
                INFERENCE_MAX_SIDE / max(frame_width, frame_height),
            )
            if scale < 1.0:
                inference_frame = cv2.resize(
                    image,
                    (
                        max(1, int(frame_width * scale)),
                        max(1, int(frame_height * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                inference_frame = image.copy()

            self._inference_future = self._inference_executor.submit(
                self._detect_and_scale,
                inference_frame,
                confidence_threshold,
                frame_width,
                frame_height,
            )
            self._inference_generation = settings_generation
            self._last_inference_at = now

        with self._settings_lock:
            detections = tuple(self._detections)
            detection_error = self._detection_error

        accepted = []
        for detection in detections:
            object_class = detection["class"]
            confidence = detection["confidence"]
            if confidence < confidence_threshold:
                continue
            if object_filter != "All Objects" and object_class != object_filter:
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
            if fresh_detections:
                self._log_detection(detection)

        if detection_error:
            status = "LIVE VIDEO · DETECTION ERROR · SEE SERVER LOGS"
        else:
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

    def _detect_and_scale(
        self,
        frame,
        confidence_threshold,
        frame_width,
        frame_height,
    ):
        detections = self.detector.detect(
            frame,
            confidence=confidence_threshold,
        )
        scale_x = frame_width / frame.shape[1]
        scale_y = frame_height / frame.shape[0]
        return [
            {
                **detection,
                "x": int(detection["x"] * scale_x),
                "y": int(detection["y"] * scale_y),
                "w": int(detection["w"] * scale_x),
                "h": int(detection["h"] * scale_y),
            }
            for detection in detections
        ]

    def _log_detection(self, detection):
        object_class = detection["class"]
        now = time.monotonic()
        if now - self.last_logged.get(object_class, 0.0) < LOG_COOLDOWN:
            return

        self.last_logged[object_class] = now
        self.log_worker.submit(detection)
