from pathlib import Path


def _creating_path_files():

    base_directory = Path('/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/images')

    train_image_directory = base_directory / 'train2017'
    test_image_directory = base_directory / 'test2017'
    val_image_directory = base_directory / 'val2017'

    output_train_file_path = train_image_directory / 'train2017.txt'
    output_test_file_path = test_image_directory / 'test-dev2017.txt'
    output_val_file_path = val_image_directory / 'val2017.txt'

    train_image_paths = list(train_image_directory.glob('*.jpg'))
    test_image_paths = list(test_image_directory.glob('*.jpg'))
    val_image_paths = list(val_image_directory.glob('*.jpg'))

    # Write to train.txt with relative paths
    with open(output_train_file_path, 'w') as file:
        for path in train_image_paths:
            # Construct relative path
            relative_path = './images/train2017/' + path.name
            file.write(relative_path + '\n')

        # Write to test.txt with relative paths
        with open(output_test_file_path, 'w') as file:
            for path in test_image_paths:
                # Construct relative path
                relative_path = './images/test2017/' + path.name
                file.write(relative_path + '\n')

        # Write to val.txt with relative paths
        with open(output_val_file_path, 'w') as file:
            for path in val_image_paths:
                # Construct relative path
                relative_path = './images/val2017/' + path.name
                file.write(relative_path + '\n')


def main():
    print('Creating files with paths')
    _creating_path_files()


if __name__ == '__main__':
    main()



