import unittest
from unittest.mock import patch

import detector as detector_module
from detector import ObjectDetector


class ObjectDetectorTests(unittest.TestCase):
    @patch("detector.YOLO")
    @patch("detector.torch.set_num_threads")
    def test_uses_a_single_cpu_thread_for_cached_model_instance(
        self,
        set_num_threads,
        yolo,
    ):
        with patch("detector.torch_utils.NUM_THREADS", 7):
            object_detector = ObjectDetector(
                model_path="yolov8n.pt",
                image_size=416,
            )
            self.assertEqual(detector_module.torch_utils.NUM_THREADS, 1)

        set_num_threads.assert_called_once_with(1)
        yolo.assert_called_once_with("yolov8n.pt")
        self.assertEqual(object_detector.image_size, 416)


if __name__ == "__main__":
    unittest.main()
