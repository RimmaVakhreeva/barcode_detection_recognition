import cv2
import os
from pycocotools.coco import COCO

# Path to the COCO annotations file
#annotation_file = 'coco/annotations/train.json'
annotation_file = '/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition/test.json'

# Path to the directory containing images
# image_directory = 'coco/images/train2017'
image_directory = '/Users/rimma_vakhreeva/PycharmProjects/barcode_detection_recognition'

# Initialize COCO api for instance annotations
coco = COCO(annotation_file)

# Load the categories or all images
cat_ids = coco.getCatIds(catNms=['person'])  # You can change 'person' to any category you're interested in
img_ids = coco.getImgIds(catIds=cat_ids)

# Loop through the images
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_directory, img['file_name'])
    image = cv2.imread(image_path)

    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    # Draw bounding boxes on the image
    for ann in annotations:
        x, y, width, height = ann['bbox']
        top_left = int(x), int(y)
        bottom_right = int(x + width), int(y + height)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue color, 2px thickness

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # Wait for a key press to move to the next image

cv2.destroyAllWindows()
