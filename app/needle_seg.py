# project_base/needle_seg.py
import numpy as np
from ultralytics import YOLO


class NeedleSegmenter:
    def __init__(self, model_path, conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_tip(self, img, center):
        """
        Возвращает ТОЛЬКО tip (x, y)
        """
        results = self.model.predict(
            img,
            imgsz=img.shape[:2],
            conf=self.conf,
            task="segment",
            verbose=False
        )

        r = results[0]
        if r.masks is None:
            return None

        masks = r.masks.data.cpu().numpy()
        if len(masks) == 0:
            return None

        # берём самую большую маску
        mask = masks[np.argmax([m.sum() for m in masks])]

        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        center = np.array(center)
        dists = np.linalg.norm(points - center, axis=1)

        tip = points[np.argmax(dists)]
        return tuple(tip)
