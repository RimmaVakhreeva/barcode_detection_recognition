import json
import pandas as pd
from pathlib import Path

train_data = pd.read_csv('/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/data/train_annotations.csv')

temp_train_data = train_data[['filename', 'width', 'height']]
train_images = []
filenames_to_idx = {}

for idx, (_, (filename, w, h)) in enumerate(temp_train_data.iterrows()):
    filenames_to_idx[filename] = idx
    train_images.append({
        'id': idx,
        'width': w,
        'height': h,
        'file_name': filename
    })

train_annotations_id = []
for idx in enumerate(train_data['filename']):
    train_annotations_id.append(idx)
dict_train_annotations_id = dict(train_annotations_id)

train_categories = [{"id": 0, "name": 'item'}]

train_annotations = []
for idx, (_, row) in enumerate(train_data.iterrows()):
    current_filename = row['filename']
    image_index = filenames_to_idx[current_filename]
    x, y, width, height = row[['x_from', 'y_from', 'width', 'height']]

train_annotations.append({
        'id': idx,
        'image_id': image_index,
        'category_id': 0,
        "area": width * height,
        "bbox": [x, y, width, height],
        "iscrowd": 0
    })


train_coco_ans = {"images": train_images, "annotations": train_annotations, 'categories': train_categories}

with open('./train.json', 'w') as f:
    json.dump(train_coco_ans, f)


