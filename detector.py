# detector.py
# YOLO Object Detection Module
# Assignment 5 - Real-Time Object Detection & Logging Platform

from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.70):
        """
        Initialize YOLO object detector.

        Parameters:
            model_path: YOLO model file
            confidence: Minimum confidence threshold
        """

        self.model = YOLO(model_path)
        self.confidence = confidence

    # -----------------------------------------------------
    # DETECT OBJECTS
    # -----------------------------------------------------
    def detect(self, frame):
        """
        Detect objects in a single video frame.

        Returns:
            List of detected objects containing:
            class, confidence, x, y, w, h
        """

        detections = []

        # Run YOLO prediction
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        # Process YOLO results
        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                # -----------------------------------------
                # BOUNDING BOX
                # -----------------------------------------
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # -----------------------------------------
                # CONFIDENCE
                # -----------------------------------------
                confidence = float(
                    box.conf[0]
                )

                # -----------------------------------------
                # CLASS ID
                # -----------------------------------------
                class_id = int(
                    box.cls[0]
                )

                # -----------------------------------------
                # CLASS NAME
                # -----------------------------------------
                object_class = self.model.names[
                    class_id
                ]

                # -----------------------------------------
                # STORE DETECTION
                # -----------------------------------------
                detection = {
                    "class": object_class,
                    "confidence": confidence,
                    "x": x1,
                    "y": y1,
                    "w": width,
                    "h": height
                }

                detections.append(detection)

        return detections

    # -----------------------------------------------------
    # CHANGE CONFIDENCE THRESHOLD
    # -----------------------------------------------------
    def set_confidence(self, confidence):
        """
        Update confidence threshold.
        """

        self.confidence = confidence