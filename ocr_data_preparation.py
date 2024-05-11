import pandas as pd
from pathlib import Path


images_path = Path("./images")
data = pd.read_csv('/Users/rimma_vakhreeva/Downloads/data/annotations.tsv', sep='\t')
data['filename'] = data['filename'].apply(lambda x: Path(x))

test_folder = images_path / 'test2017'

test_images = []
codes = []
x_from_bbox = []
y_from_bbox = []
width_bbox = []
height_bbox = []

for index, row in data.iterrows():
    for img in test_folder.iterdir():
        if img.suffix in ['.jpg', '.png', '.jpeg'] and row['filename'].name == img.name:
            test_images.append(str(img))  # store string representation of path
            codes.append(row['code'])
            x_from_bbox.append(row['x_from'])
            y_from_bbox.append(row['y_from'])
            width_bbox.append(row['width'])
            height_bbox.append(row['height'])

df = pd.DataFrame({
    'filename': test_images,
    'code': codes,
    'x_from': x_from_bbox,
    'y_from': y_from_bbox,
    'width': width_bbox,
    'height': height_bbox
})

df.to_csv('ocr_annotations.csv', index=False)
