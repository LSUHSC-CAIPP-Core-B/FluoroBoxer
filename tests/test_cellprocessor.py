import unittest

import numpy as np

try:
    from CellProcessor.CellProcessor import crop_img_from_label, get_label_yolo, yolo_to_original
except ImportError as exc:  # pragma: no cover - environment-dependent OpenCV import
    crop_img_from_label = get_label_yolo = yolo_to_original = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"OpenCV dependency unavailable: {IMPORT_ERROR}")
class TestCellProcessor(unittest.TestCase):
    def test_yolo_roundtrip_dimensions(self):
        start_pt = (10, 20)
        end_pt = (30, 40)
        img_w, img_h = 100, 80

        yolo_label = get_label_yolo(start_pt, end_pt, img_w, img_h)

        img = np.zeros((img_h, img_w), dtype=np.uint8)
        x, y, w, h = yolo_to_original(img, yolo_label)

        self.assertEqual((x, y, w, h), (10, 20, 20, 20))

    def test_crop_img_from_label_generates_mask_with_matching_shape(self):
        img = np.zeros((50, 30), dtype=np.uint8)
        yolo_label = (0.5, 0.5, 0.4, 0.4)

        crop, mask = crop_img_from_label(img, yolo_label, withMask=True)

        self.assertEqual(crop.shape, mask.shape)


if __name__ == "__main__":
    unittest.main()
