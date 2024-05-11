from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from ocr.ocr_train_project.crnn import CRNN


class OcrModel:
    def __init__(self, weights: Path):
        assert weights.exists()
        self.model = CRNN(
            cnn_backbone_name='resnet18d',
            cnn_backbone_pretrained=False,
            cnn_output_size=4608,
            rnn_features_num=128,
            rnn_dropout=0.1,
            rnn_bidirectional=True,
            rnn_num_layers=2,
            num_classes=11
        )
        self.h, self.w = 280, 523
        self.model.load_state_dict(torch.load(str(weights), map_location='cpu'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab = '0123456789'
        self.index2char = {idx + 1: char for idx, char in enumerate(self.vocab)}
        self.index2char[0] = " "

    def recognize(self, image: np.ndarray, bboxes: List[np.ndarray]) -> List[str]:
        texts = []
        for (x1, y1, x2, y2, conf) in bboxes:
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            crop = cv2.resize(crop, (self.w, self.h))
            crop = np.ascontiguousarray(crop.transpose((2, 0, 1))[::-1] / 255.0)[None, ...]
            pred = self.model(torch.tensor(crop, dtype=torch.float32, device=self.device))
            text = self.model.decode_output(pred, vocab=self.vocab)[0]
            texts.append(text)
        return texts


if __name__ == "__main__":
    image_path = Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/"
                      "images/test2017/0f61e632-59a2-4bc8-9119-24178ca64752--ru.8487fa87-b8cd-4def-9e54-5cf42bc4148e.jpg")
    image = cv2.imread(str(image_path))
    bboxes = [np.array([116.0, 499.0, 773.0, 765.0, 1.0])]

    model = OcrModel(Path("/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/crnn_last.pt"))
    print(model.recognize(image, bboxes))
