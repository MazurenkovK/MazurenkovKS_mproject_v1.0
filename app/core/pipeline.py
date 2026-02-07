# app/core/pipeline.py
import numpy as np

from detectors.meter_detector import MeterDetector
from detectors.yolo_detect import GaugeYOLO
from segmenters.needle_seg import NeedleSegmenter
from geometry.geometry import angle_from_minimum, normalize_angle
from ocr.scale_ocr import ScaleOCR

import cv2
import numpy as np

class PressurePipeline:
    def __init__(
        self,
        gauge_model_path,
        seg_model_path,
        pressure_min=0.0,
        pressure_max=6.0
    ):
        self.meter = MeterDetector()
        self.gauge = GaugeYOLO(gauge_model_path)
        self.needle = NeedleSegmenter(seg_model_path)
        self.ocr = ScaleOCR(debug=True)

        self.p_min = pressure_min
        self.p_max_default = pressure_max

    def process(self, img, visualize=True):
        # STEP 1 — meter
        cropped, meter_center = self.meter.detect_and_crop(img)
        if cropped is None:
            raise RuntimeError("Meter not found")

        # STEP 2 — scale
        minimum, maximum, img_yolo = self.gauge.detect_scale(cropped)
        if minimum is None or maximum is None:
            raise RuntimeError("Scale not detected")
        
        # OCR MAX
        max_value = self.ocr.detect_max_value(img_yolo, maximum)
        print("OCR max value:", max_value)
        if max_value is not None and max_value < 20:
            p_max = max_value
        else:
            p_max = self.p_max_default

        # Получаем размеры изображения
        height, width = img_yolo.shape[:2]

        # Определяем центр изображения
        center = (width // 2, height // 2)

        # STEP 3 — needle
        tip = self.needle.detect_tip(img_yolo, center)
        if tip is None:
            raise RuntimeError("Needle not detected")
        print(f"tip: {tip}, center: {center}, min: {minimum}, max: {maximum}")
        # STEP 4 — angles
        angle_tip = angle_from_minimum(tip, center, minimum)
        angle_max = angle_from_minimum(maximum, center, minimum)

        print(f"angle_tip={angle_tip:.1f}, angle_max={angle_max:.1f}")

        value = normalize_angle(angle_tip, angle_max)
        if value is None:
            print("⚠️ Needle outside scale")
            print(f"status: OUT_OF_SCALE, angle_tip={angle_tip}, angle_max={angle_max}")
            if visualize:
                self._visualize(img_yolo, center, minimum, maximum, tip,
                                angle_tip, angle_max, None)
            return None

        # STEP 5 — pressure
        pressure = self.p_min + value * (p_max - self.p_min)

        if visualize:
            self._visualize(img_yolo, center, minimum, maximum, tip,
                            angle_tip, angle_max, pressure)

        return pressure

    # -------------------------------------------------

    def _visualize(
        self,
        img,
        center,
        minimum,
        maximum,
        tip,
        angle_tip,
        angle_max,
        pressure
    ):
        vis = img.copy()

        center  = tuple(map(int, center))
        minimum = tuple(map(int, minimum))
        maximum = tuple(map(int, maximum))
        tip     = tuple(map(int, tip))

        # points
        cv2.circle(vis, center, 8, (0, 255, 255), -1)   # center — yellow
        cv2.circle(vis, minimum, 8, (0, 255, 0), -1)   # minimum — green
        cv2.circle(vis, maximum, 8, (0, 0, 255), -1)   # maximum — red
        cv2.circle(vis, tip, 8, (255, 0, 255), -1)     # tip — magenta

        # rays
        cv2.line(vis, center, minimum, (0, 255, 0), 2)
        cv2.line(vis, center, maximum, (0, 0, 255), 2)
        cv2.line(vis, center, tip, (255, 0, 255), 3)

        # text
        cv2.putText(
            vis,
            f"angle_tip: {angle_tip:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        cv2.putText(
            vis,
            f"angle_max: {angle_max:.1f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if pressure is not None:
            cv2.putText(
                vis,
                f"PRESSURE: {pressure:.2f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        cv2.imshow("Pressure pipeline debug", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
