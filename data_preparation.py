from pathlib import Path
import random
import shutil
import pandas as pd


def main():
    root = Path('/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition')

    source_folder = root / 'data' / 'images'
    destination_folder = root / 'data' / 'test_images'

    destination_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(source_folder.glob('*.jpg'))

    test_images = random.sample(image_files, 50)

    for image in test_images:
        shutil.move(str(image), destination_folder / image.name)

    annotations = root / 'data' / 'annotations.tsv'

    if annotations.exists():
        data = pd.read_csv(annotations, sep='\t')

        data['filename'] = data['filename'].apply(lambda x: Path(x).name)

        test_data = data[data['filename'].isin([image.name for image in test_images])]
        train_data = data[~data['filename'].isin([image.name for image in test_images])]

        test_data.to_csv(root / 'data' / 'test_annotations.csv', index=False)
        train_data.to_csv(root / 'data' / 'train_annotations.csv', index=False)


if __name__ == '__main__':
    main()



