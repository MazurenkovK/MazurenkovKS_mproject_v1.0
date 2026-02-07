# app/ocr/scale_ocr.py
import cv2
import numpy as np
import pytesseract
import re

class ScaleOCR:
    def __init__(self, debug=False):
        self.debug = debug
        self.config = (
            "--psm 6 -c tessedit_char_whitelist=0123456789.,"
        )
        self.resize = 2
        

    def detect_max_value(self, img, maximum_point, roi_size=70):
        """
        img            — img_yolo (BGR)
        maximum_point  — (x, y) точка максимума шкалы
        """
        h, w = img.shape[:2]
        x, y = maximum_point

        # ROI справа-низу от maximum
        half = roi_size // 2
        x1 = max(0, x - half-20)
        y1 = max(0, y - half-20)
        x2 = min(w, x + half-20)
        y2 = min(h, y + half-20)
 

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print("OCR ROI is empty")
            return None
        roi = cv2.resize(roi, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_CUBIC)
        # Препроцессинг
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            41, 15
        )

        kernel = np.ones((5, 5), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        text = pytesseract.image_to_string(
            bin_img,
            config=self.config
        )
        numbers = re.findall(r"\d+[.,]?\d*", text)
        numbers = [float(n.replace(",", ".")) for n in numbers]

        max_value = max(numbers) if numbers else None

        if self.debug:
            self._visualize(img, roi, (x1, y1, x2, y2), maximum_point, max_value, bin_img)

        return max_value

    def _visualize(self, img, roi, roi_box, max_point, value, bin_img):
        vis = img.copy()
        x1, y1, x2, y2 = roi_box

        # ROI box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Maximum point
        cv2.circle(vis, max_point, 6, (0, 0, 255), -1)

        # Text
        label = f"MAX OCR: {value}" if value is not None else "MAX OCR: None"
        cv2.putText(
            vis, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        # cv2.imshow("Scale OCR Visualization", vis)
        # cv2.imshow("OCR ROI", roi)
        cv2.imshow("OCR BIN", bin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
