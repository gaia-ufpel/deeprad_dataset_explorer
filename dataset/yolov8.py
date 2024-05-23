import os
import glob
import yaml
from PIL import Image

from dataset import Dataset, Point, SegmentationAnnotation

class YOLOv8Adapter:
    @staticmethod
    def load(yaml_path: str) -> Dataset:
        """
        TODO: Add documentation
        """
        yolov8_yaml = None
        with open(yaml_path, 'r') as reader:
            yolov8_yaml = yaml.safe_load(reader)

        id2class = yolov8_yaml['names']
        image_id2image_name = {}
        image_id2image_dimensions = {}
        annotation_id2image_id = {}
        annotation_id2annotation = {}

        for image_path in glob.glob(os.path.join(yolov8_yaml['path'], '*.jpg')):
            image_name = os.path.basename(image_path)
            image_id = len(image_id2image_name)
            image_id2image_name.update({
                image_id: image_name
            })

            image = Image.open(image_path)
            image_id2image_dimensions.update({
                image_id: image.size
            })

            txt_file_path = image_path.replace('.jpg', '.txt')
            if not os.path.exists(txt_file_path):
                continue

            txt_file = None
            with open(txt_file_path, 'r') as reader:
                txt_file = reader.readlines()

            for line in txt_file:
                class_id, *poly = line.split(' ')
                poly = list(map(float, poly))
                poly = [int(poly[i] * image.size[i % 2]) for i in range(len(poly))]

                x_values = [poly[i] for i in range(0, len(poly), 2)]
                y_values = [poly[i] for i in range(1, len(poly), 2)]
                x = min(x_values)
                y = min(y_values)
                width = max(x_values) - x
                height = max(y_values) - y

                annotation = SegmentationAnnotation(
                    class_id = class_id,
                    x = x,
                    y = y,
                    width = width,
                    height = height,
                    points = [Point(poly[i], poly[i+1]) for i in range(4, len(poly), 2)]
                )

                annotation_id = len(annotation_id2image_id)
                annotation_id2image_id.update({
                    annotation_id: image_id
                })
                annotation_id2annotation.update({
                    annotation_id: annotation
                })

        return Dataset(
            id2class = id2class,
            data_path = yolov8_yaml['path'],
            image_id2image_name = image_id2image_name,
            image_id2image_dimensions = image_id2image_dimensions,
            annotation_id2image_id = annotation_id2image_id,
            annotation_id2annotation = annotation_id2annotation
        )

    @staticmethod
    def save(dataset: Dataset, json_path: str) -> None:
        pass