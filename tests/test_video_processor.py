import unittest
import time
from unittest.mock import patch

import av
import numpy as np

from video_processor import LiveDetectionProcessor


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.calls = []

    def detect(self, frame, confidence):
        self.calls.append((frame.shape, confidence))
        return self.detections


class FakeLogWorker:
    def __init__(self):
        self.events = []

    def submit(self, detection):
        self.events.append(detection)


class LiveDetectionProcessorTests(unittest.TestCase):
    def setUp(self):
        self.frame = av.VideoFrame.from_ndarray(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            format="bgr24",
        )

    def test_returns_every_frame_but_caps_inference_rate(self):
        detector = FakeDetector()
        processor = LiveDetectionProcessor(
            detector,
            FakeLogWorker(),
            confidence_threshold=0.70,
            object_filter="All Objects",
        )
        timestamps = [100.0 + index / 30 for index in range(30)]

        with patch("video_processor.time.monotonic", side_effect=timestamps):
            output_frames = []
            for _ in timestamps:
                output_frames.append(processor.recv(self.frame))
                time.sleep(0.002)
        processor.on_ended()

        self.assertEqual(len(output_frames), 30)
        self.assertTrue(all(isinstance(frame, av.VideoFrame) for frame in output_frames))
        self.assertGreaterEqual(len(detector.calls), 5)
        self.assertLessEqual(len(detector.calls), 10)
        self.assertEqual(detector.calls[0][0], (360, 640, 3))

    def test_filters_and_rescales_detections_before_overlay_and_logging(self):
        detector = FakeDetector(
            [
                {
                    "class": "person",
                    "confidence": 0.82,
                    "x": 10,
                    "y": 20,
                    "w": 30,
                    "h": 40,
                },
                {
                    "class": "cell phone",
                    "confidence": 0.95,
                    "x": 1,
                    "y": 2,
                    "w": 3,
                    "h": 4,
                },
                {
                    "class": "person",
                    "confidence": 0.69,
                    "x": 1,
                    "y": 2,
                    "w": 3,
                    "h": 4,
                },
            ]
        )
        worker = FakeLogWorker()
        processor = LiveDetectionProcessor(
            detector,
            worker,
            confidence_threshold=0.70,
            object_filter="person",
        )

        with patch(
            "video_processor.time.monotonic",
            side_effect=[100.0, 100.01, 100.01],
        ):
            processor.recv(self.frame)
            processor._inference_future.result(timeout=5)
            output = processor.recv(self.frame)
        processor.on_ended()

        self.assertEqual(
            [(event["class"], event["x"], event["y"], event["w"], event["h"])
             for event in worker.events],
            [("person", 20, 40, 60, 80)],
        )
        self.assertIsInstance(output, av.VideoFrame)

    def test_detection_log_cooldown_is_applied_per_class(self):
        worker = FakeLogWorker()
        processor = LiveDetectionProcessor(
            FakeDetector(),
            worker,
            confidence_threshold=0.70,
            object_filter="All Objects",
        )
        detection = {
            "class": "person",
            "confidence": 0.82,
            "x": 10,
            "y": 20,
            "w": 30,
            "h": 40,
        }

        with patch(
            "video_processor.time.monotonic",
            side_effect=[100.0, 101.9, 102.1],
        ):
            processor._log_detection(detection)
            processor._log_detection(detection)
            processor._log_detection(detection)
        processor.on_ended()

        self.assertEqual(len(worker.events), 2)


if __name__ == "__main__":
    unittest.main()
