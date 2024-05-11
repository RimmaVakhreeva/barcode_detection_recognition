import random
from pathlib import Path

import barcode  # pip install git+https://github.com/WhyNotHugo/python-barcode
import cv2
import numpy as np
import torch
import albumentations as alb

from barcode.writer import ImageWriter
from ocr.ocr_train_project.generator.src.generator import overlay


class BarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size, vocab):
        self.epoch_size = epoch_size
        self.char2index = dict((char, vocab.index(char) + 1) for char in vocab)

        background_parent_path = Path(__file__).parent / "backgrounds"
        self.backgrounds = [
            background_parent_path / "background_0.jpg",
            background_parent_path / "background_1.jpg",
            background_parent_path / "background_2.jpg",
        ]

    def __getitem__(self, i: int):
        value = str(np.random.randint(10e11, 10e12))
        ean = barcode.EAN13(value, writer=ImageWriter(), guardbar=True)
        value_w_checksum = ean.get_fullcode()
        value_w_checksum = value_w_checksum.replace(" ", "")
        rendered_barcode = ean.render()
        rendered_barcode = 255 - cv2.cvtColor(np.asarray(rendered_barcode), cv2.COLOR_RGB2GRAY)
        h, w = rendered_barcode.shape
        barcode_image = np.zeros((h, w, 3), dtype=np.uint8)
        background = cv2.imread(str(random.choice(self.backgrounds)))
        barcode_img = overlay(
            background,
            barcode_image,
            rendered_barcode,
            transforms=alb.Compose([
                alb.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=(-13, 13),
                                     border_mode=cv2.BORDER_CONSTANT, p=1),
                alb.GaussNoise(),
                alb.OneOf([
                    alb.MotionBlur(blur_limit=11, p=0.4),
                    alb.MedianBlur(blur_limit=5, p=0.4),
                    alb.Blur(blur_limit=11, p=0.4),
                ], p=0.5),
                alb.OneOf([
                    alb.OpticalDistortion(p=0.4, border_mode=cv2.BORDER_CONSTANT),
                    alb.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT),
                ], p=0.4),
                alb.OneOf([
                    alb.CLAHE(clip_limit=2),
                    alb.RandomBrightnessContrast(),
                ], p=0.2),
                alb.HueSaturationValue(p=0.3),
                alb.RGBShift(r_shift_limit=(-20, 20))
            ])
        )

        image = torch.FloatTensor(barcode_img / 255.0).permute(2, 0, 1)
        return (
            image,
            torch.LongTensor(self.encode_value(value_w_checksum)),
            torch.LongTensor([len(value_w_checksum)]),
            value_w_checksum
        )

    def __len__(self):
        return self.epoch_size

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)


class TestBarcodeDataset(torch.utils.data.Dataset):
    def __init__(self, directory, vocab):
        self.directory = Path(directory)
        self.image_files = list(self.directory.glob('*.jpg'))
        self.h, self.w = 280, 523
        self.vocab = vocab
        self.char2index = dict((char, vocab.index(char) + 1) for char in vocab)

    def __getitem__(self, i: int):
        img_path = self.image_files[i]
        image = cv2.imread(str(img_path))
        image = cv2.resize(image, (self.w, self.h))
        image = torch.FloatTensor(image / 255.0).permute(2, 0, 1)
        return (
            image,
            torch.LongTensor(self.encode_value(img_path.stem)),
            torch.LongTensor([len(img_path.stem)]),
            img_path.stem
        )

    def __len__(self):
        return len(self.image_files)

    def encode_value(self, value):
        return [self.char2index[char] for char in value]

    def decode_predict(self, predict):
        return ''.join(self.vocab[index - 1] for index in predict)


if __name__ == "__main__":
    ds = BarcodeDataset(epoch_size=1024, vocab="0123456789")
    for idx in range(1024):
        _ = ds[idx]
