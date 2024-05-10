import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov9"  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)


class Yolov9:
    def __init__(self, weights: Path, device: str = "cpu"):
        assert weights.exists()
        self.model = DetectMultiBackend(weights, device=device)
        self.model_stride = self.model.stride
        self.is_torch_model = self.model.pt
        self.source = 'images/test2017'
        self.img_size = (640, 640)
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 1000

    def detect_barcodes(self, image: np.ndarray) -> List[np.ndarray]:
        augmented_image = letterbox(image, self.img_size, stride=self.model_stride, auto=self.is_torch_model)[0]
        augmented_image = augmented_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        augmented_image = np.ascontiguousarray(augmented_image)

        augmented_image = torch.from_numpy(augmented_image).to(self.model.device)
        augmented_image = augmented_image.half() if self.model.fp16 else augmented_image.float()  # uint8 to fp16/32
        augmented_image /= 255  # 0 - 255 to 0.0 - 1.0
        augmented_image = augmented_image[None, ...]

        pred = self.model(augmented_image, augment=False, visualize=False)
        pred = pred[0][1]

        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=None,
            agnostic=self.agnostic_nms,
            max_det=self.max_det
        )
        for i, det in enumerate(pred):
            det[:, :4] = scale_boxes(augmented_image.shape[2:], det[:, :4], image.shape).round()
        return [bbox.cpu().numpy()[0, :5] for bbox in pred]


if __name__ == "__main__":
    weights = Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/best.pt")
    images_path = Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/images/test2017")

    yolov9 = Yolov9(weights)
    for image_path in images_path.glob('*'):
        if image_path.suffix.lower() in ['.jpg']:
            image = cv2.imread(str(image_path))
            if image is not None:
                results = yolov9.detect_barcodes(image)
                for (x1, y1, x2, y2, conf) in results:
                    cv2.rectangle(image,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (255, 0, 0), thickness=2,
                                  lineType=cv2.LINE_AA)
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
            else:
                print(f"Failed to load {image_path.name}")
        else:
            continue
