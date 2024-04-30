import json
from pathlib import Path

root = Path('/Users/rimma_vakhreeva/PycharmProjects/yolov9/coco')
coco_annotations = root / 'annotations/instances_val2017.json'

with open(coco_annotations, 'r') as file:
    data = json.load(file)

print(data.keys())
print(len(data['annotations']))