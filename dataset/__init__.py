import os
import cv2
import yaml
import json
import glob
import math
import shutil
import random
import numpy as np
from PIL import Image
from logging import Logger
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

class Annotation:
    pass

@dataclass
class ClassificationAnnotation(Annotation):
    class_id: int

@dataclass
class DetectionAnnotation(ClassificationAnnotation):
    x: float
    y: float
    width: float
    height: float

@dataclass
class SegmentationAnnotation(DetectionAnnotation):
    points: list[Point]

class Dataset:
    """
    TODO: Add documentation
    """
    def __init__(self,
                id2class: dict[int, str], 
                data_path: str, 
                image_id2image_name: dict[int, str], 
                image_id2image_dimensions: dict[int, tuple[int, int]], 
                annotation_id2image_id: dict[int, int], 
                annotation_id2annotation: dict[int, list[Annotation]]
                ):
        """
        TODO: Add documentation
        """
        
        self.id2class = id2class
        self.class2id = {v: k for k, v in id2class.items()}
        self.data_path = data_path

        self.image_id2image_name: dict[int, str] = image_id2image_name
        self.image_path2image_id: dict[str, int] = {v: k for k, v in image_id2image_name.items()}
        self.image_id2image_dimensions: dict[int, tuple[int, int]] = image_id2image_dimensions
        self.annotation_id2image_id: dict[int, int] = annotation_id2image_id
        self.image_id2annotation_ids: dict[int, list[int]] = {image_id: [] for image_id in image_id2image_name.keys()}
        for annotation_id, image_id in annotation_id2image_id.items():
            self.image_id2annotation_ids[image_id].append(annotation_id)
        self.annotation_id2annotation: dict[int, list[Annotation]] = annotation_id2annotation

    def __len__(self):
        """
        TODO: Add documentation
        """
        return len(self.image_id2image_name)
    
    def __getitem__(self, image_id: int):
        """
        TODO: Add documentation
        """
        return np.array(Image.open(os.path.join(self.data_path, self.image_id2image_name[image_id])))
    
    def __iter__(self):
        """
        TODO: Add documentation
        """
        return iter(self.image_id2image_name.keys())
    
    def get_image(self, image_id: int):
        return np.array(Image.open(os.path.join(self.data_path, self.image_id2image_name[image_id])))

    def get_annotation(self, annotation_id: int):
        return self.annotation_id2annotation[annotation_id]

    def count_classe_instances(self):
        """
        TODO: Add documentation
        """
        class_instances = {class_id: 0 for class_id in self.id2class.keys()}
        for annotation in self.annotation_id2annotation.values():
            class_instances[annotation.class_id] += 1
        return class_instances

    def split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """
        TODO: Add documentation
        """
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("The sum of the ratios must be 1")

        image_ids = self.image_id2image_name.keys()
        random.shuffle(image_ids)

        train_image_ids = image_ids[: int(len(image_ids) * train_ratio)]
        val_image_ids = image_ids[int(len(image_ids) * train_ratio) : int(len(image_ids) * val_ratio)]
        test_image_ids = image_ids[int(len(image_ids) * (train_ratio + val_ratio)) :]

        train_image_id2image_name = {}
        train_image_id2image_dimensions = {}
        train_annotation_id2image_id = {}
        train_annotation_id2annotation = {}
        for image_id in train_image_ids:
            train_image_id2image_name.update({
                image_id: self.image_id2image_name[image_id]
            })
            train_image_id2image_dimensions.update({
                image_id: self.image_id2image_dimensions[image_id]
            })
            for annotation_id in self.image_id2annotation_ids[image_id]:
                train_annotation_id2image_id.update({
                    annotation_id: image_id
                })
                train_annotation_id2annotation.update({
                    annotation_id: self.annotation_id2annotation[annotation_id]
                })

        val_image_id2image_name = {}
        val_image_id2image_dimensions = {}
        val_annotation_id2image_id = {}
        val_annotation_id2annotation = {}
        for image_id in val_image_ids:
            val_image_id2image_name.update({
                image_id: self.image_id2image_name[image_id]
            })
            val_image_id2image_dimensions.update({
                image_id: self.image_id2image_dimensions[image_id]
            })
            for annotation_id in self.image_id2annotation_ids[image_id]:
                val_annotation_id2image_id.update({
                    annotation_id: image_id
                })
                val_annotation_id2annotation.update({
                    annotation_id: self.annotation_id2annotation[annotation_id]
                })

        test_image_id2image_name = {}
        test_image_id2image_dimensions = {}
        test_annotation_id2image_id = {}
        test_annotation_id2annotation = {}
        for image_id in test_image_ids:
            test_image_id2image_name.update({
                image_id: self.image_id2image_name[image_id]
            })
            test_image_id2image_dimensions.update({
                image_id: self.image_id2image_dimensions[image_id]
            })
            for annotation_id in self.image_id2annotation_ids[image_id]:
                test_annotation_id2image_id.update({
                    annotation_id: image_id
                })
                test_annotation_id2annotation.update({
                    annotation_id: self.annotation_id2annotation[annotation_id]
                })

        train_ds = Dataset(
            id2class = self.id2class,
            data_path = self.data_path,
            image_id2image_name = train_image_id2image_name,
            image_id2image_dimensions = train_image_id2image_name,
            annotation_id2image_id = train_annotation_id2image_id,
            annotation_id2annotation = train_annotation_id2annotation
        )
        val_ds = Dataset(
            id2class = self.id2class,
            data_path = self.data_path,
            image_id2image_name = val_image_id2image_name,
            image_id2image_dimensions = val_image_id2image_name,
            annotation_id2image_id = val_annotation_id2image_id,
            annotation_id2annotation = val_annotation_id2annotation
        )
        test_ds = Dataset(
            id2class = self.id2class,
            data_path = self.data_path,
            image_id2image_name = test_image_id2image_name,
            image_id2image_dimensions = test_image_id2image_dimensions,
            annotation_id2image_id = test_annotation_id2image_id,
            annotation_id2annotation = test_annotation_id2annotation
        )

        return train_ds, val_ds, test_ds

    def copy_dataset(self, destination_path: str):
        """
        TODO: Add documentation
        """
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        shutil.copy(self.data_path, destination_path)

        return Dataset(
            self.id2class,
            destination_path,
            self.image_id2image_name,
            self.image_id2image_dimensions,
            self.annotation_id2image_id,
            self.annotation_id2annotation
        )

    def crop_images(self, x: int, y: int, width: int, height: int):
        """
        TODO: Add documentation
        """
        for image_id, image_name in self.image_id2image_name.items():
            image = Image.open(os.path.join(self.data_path, image_name))
            image = image.crop((x, y, x + width, y + height))
            image.save(os.path.join(self.data_path, image_name))

            for annotation_id in self.image_id2annotation_ids[image_id]:
                annotation = self.annotation_id2annotation[annotation_id]

                annotation.x -= x
                annotation.y -= y

                for point in annotation.points:
                    point.x -= x
                    point.y -= y

    def crop_images_by_annotations(self, margin: int):
        """
        TODO: Add documentation
        """
        for image_id, image_path in self.image_id2image_name.items():
            image = Image.open(image_path)
            annotations_ids = self.image_id2annotation_ids[image_id]

            x_values = []
            y_values = []
            for annotation_id in annotations_ids:
                annotation = self.annotation_id2annotation[annotation_id]
                x_values.append(annotation.x)
                x_values.append(annotation.x + annotation.width)
                y_values.append(annotation.y)
                y_values.append(annotation.y + annotation.height)

            x_min = min(x_values) - margin
            if x_min < 0:
                x_min = 0
            x_max = max(x_values) + margin
            if x_max > self.image_id2image_dimensions[image_id][0]:
                x_max = self.image_id2image_dimensions[image_id][0]
            y_min = min(y_values) - margin
            if y_min < 0:
                y_min = 0
            y_max = max(y_values) + margin
            if y_max > self.image_id2image_dimensions[image_id][1]:
                y_max = self.image_id2image_dimensions[image_id][1]

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image.save(image_path)

            for annotation_id in annotations_ids:
                annotation = self.annotation_id2annotation[annotation_id]

                annotation.x -= x_min
                annotation.y -= y_min
                annotation.width = x_max - x_min
                annotation.height = y_max - y_min

                for point in annotation.points:
                    point.x -= x_min
                    point.y -= y_min

    @staticmethod
    def from_coco(coco_json_path: str, images_path: str):
        """
        TODO: Add documentation
        """
        coco_dataset = None
        with open(coco_json_path, 'r') as reader:
            coco_dataset = json.loads(reader.read())

        id2class = {}
        for class_info in coco_dataset['categories']:
            id2class.update({
                class_info['id']:class_info['name']
            })

        image_id2image_name = {}
        image_id2image_dimensions = {}
        for image_info in coco_dataset['images']:
            image_id2image_name.update({
                image_info['id']: image_info['file_name']
            })
            image_id2image_dimensions.update({
                image_info['id']: (image_info['width'], image_info['height'])
            })

        annotation_id2image_id = {}
        annotation_id2annotation = {}
        for annotation_info in coco_dataset['annotations']:
            annotation_id2image_id.update({
                annotation_info['id']: annotation_info['image_id']
            })

            annotation = SegmentationAnnotation(
                class_id = annotation_info['category_id'],
                x = annotation_info['bbox'][0],
                y = annotation_info['bbox'][1],
                width = annotation_info['bbox'][2],
                height = annotation_info['bbox'][3],
                points = [Point(annotation_info['segmentation'][i], annotation_info['segmentation'][i+1]) for i in range(0, len(annotation_info['segmentation']), 2)]
            )
            annotation_id2annotation.update({
                annotation_info['id']: annotation
            })

        return Dataset(
            id2class = id2class,
            data_path = images_path,
            image_id2image_name = image_id2image_name,
            image_id2image_dimensions = image_id2image_dimensions,
            annotation_id2image_id = annotation_id2image_id,
            annotation_id2annotation = annotation_id2annotation
        )
    
    @staticmethod
    def from_yolov8(yolov8_yaml_path: str):
        """
        TODO: Add documentation
        """
        yolov8_yaml = None
        with open(yolov8_yaml_path, 'r') as reader:
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
    def from_instance_segmentation_masks(images_path: str, masks_path: str):
        """
        TODO: Add documentation
        """
        pass

    @staticmethod
    def from_semantic_segmentation_masks(images_path: str, masks_path: str):
        """
        TODO: Add documentation
        """
        pass

    def to_coco(self, coco_json_path: str):
        """
        TODO: Add documentation
        """
        pass

    def to_yolov8(self, yolov8_yaml_path: str):
        """
        TODO: Add documentation
        """
        pass

    def to_instance_segmentation_masks(self, images_path: str, masks_path: str):
        """
        TODO: Add documentation
        """
        pass

    def to_semantic_segmentation_masks(self, images_path: str, masks_path: str):
        """
        TODO: Add documentation
        """
        pass