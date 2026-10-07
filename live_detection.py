import argparse

import cv2

from config import CAMERA_INDEX, DEFAULT_CONFIDENCE, MODEL_PATH, OBJECT_CLASSES
from detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on a webcam.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--object",
        choices=OBJECT_CLASSES,
        default="All Objects",
        dest="object_filter",
    )
    args = parser.parse_args()

    detector = ObjectDetector(model_path=MODEL_PATH, confidence=args.confidence)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check permissions or select another camera."
        )

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("Failed to read a frame from the camera.")

            for detection in detector.detect(frame):
                if (
                    args.object_filter != "All Objects"
                    and detection["class"] != args.object_filter
                ):
                    continue

                x, y = detection["x"], detection["y"]
                width, height = detection["w"], detection["h"]
                label = (
                    f'{detection["class"]}: '
                    f'{detection["confidence"] * 100:.1f}%'
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Real-Time Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()