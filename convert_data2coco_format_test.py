import json
import pandas as pd
from pathlib import Path

test_data = pd.read_csv('/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/data/test_annotations.csv')

temp_test_data = test_data[['filename', 'width', 'height']]
test_images = []
filenames_to_idx = {}

for idx, (_, (filename, w, h)) in enumerate(temp_test_data.iterrows()):
    filenames_to_idx[filename] = idx
    test_images.append({
        'id': idx,
        'width': w,
        'height': h,
        'file_name': filename
    })

test_annotations_id = []
for idx in enumerate(test_data['filename']):
    test_annotations_id.append(idx)
dict_test_annotations_id = dict(test_annotations_id)

test_categories = [{"id": 0, "name": 'item'}]

test_annotations = []
for idx, (_, row) in enumerate(test_data.iterrows()):
    current_filename = row['filename']
    image_index = filenames_to_idx[current_filename]
    x, y, width, height = row[['x_from', 'y_from', 'width', 'height']]

test_annotations.append({
        'id': idx,
        'image_id': image_index,
        'category_id': 0,
        "area": width * height,
        "bbox": [x, y, width, height],
        "iscrowd": 0
    })


test_coco_ans = {"images": test_images, "annotations": test_annotations, 'categories': test_categories}

with open('./test.json', 'w') as f:
    json.dump(test_coco_ans, f)