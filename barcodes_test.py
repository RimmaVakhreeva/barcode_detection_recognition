import pandas as pd
from pathlib import Path
import cv2


images_path = Path("./images")
cropped_images_path = images_path / 'cropped_test_images'
data = pd.read_csv('ocr_annotations.csv')

data['filename'] = data['filename'].apply(lambda x: Path(x))

test_folder = images_path / 'test2017'

for index, row in data.iterrows():
    for img_path in test_folder.iterdir():
        if img_path.suffix in ['.jpg', '.png', '.jpeg'] and row['filename'].name == img_path.name:
            img = cv2.imread(str(img_path))
            assert img is not None
            x1 = row['y_from']
            y1 = row['x_from']
            x2 = x1 + row['height']
            y2 = y1 + row['width']
            crop = img[int(y1):int(y2), int(x1):int(x2)]

            crop_image_path = cropped_images_path / f"{row['code']}.jpg"
            cv2.imwrite(str(crop_image_path), crop)



