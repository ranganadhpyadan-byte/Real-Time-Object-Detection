import threading
import unittest

from database import DetectionLogWorker


class FakeDatabase:
    def __init__(self):
        self.inserted = threading.Event()
        self.events = []

    def insert_detection(self, **event):
        self.events.append(event)
        self.inserted.set()


class DetectionLogWorkerTests(unittest.TestCase):
    def test_submitted_event_is_written_by_background_worker(self):
        database = FakeDatabase()
        worker = DetectionLogWorker(
            database_factory=lambda: database,
            max_pending=1,
        )
        detection = {
            "class": "person",
            "confidence": 0.82,
            "x": 10,
            "y": 20,
            "w": 30,
            "h": 40,
        }

        worker.submit(detection)

        self.assertTrue(database.inserted.wait(timeout=2))
        self.assertEqual(
            database.events,
            [
                {
                    "object_class": "person",
                    "confidence": 0.82,
                    "bbox_x": 10,
                    "bbox_y": 20,
                    "bbox_w": 30,
                    "bbox_h": 40,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
