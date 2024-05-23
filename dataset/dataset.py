import os
import cv2
import yaml
import json
import glob
import math
import shutil
import random
import datetime
import numpy as np
from PIL import Image
from logging import Logger
from dataclasses import dataclass
from pycocotools import mask as mask_utils

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
    area: float

    def get_mask(self, image_dimensions: tuple[int, int]) -> np.array:
        mask = np.zeros(image_dimensions, dtype=np.uint8)
        mask[int(self.y):int(self.y + self.height), int(self.x):int(self.x + self.width)] = 1
        return mask

    def compute_area_from_dimensions(self) -> None:
        self.area = self.width * self.height

@dataclass
class SegmentationAnnotation(DetectionAnnotation):
    points: list[Point]

    def get_mask(self, image_dimensions: tuple[int, int]) -> np.array:
        mask = np.zeros(image_dimensions, dtype=np.uint8)
        points = [[(point.x, point.y)] for point in self.points]
        cv2.fillPoly(mask, points, 1)
        return mask
    
    def compute_area_from_points(self) -> None:
        self.area = cv2.contourArea(np.array([[point.x, point.y] for point in self.points]))

@dataclass
class ImageInfo:
    id: int
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]

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

        self.image_id2image_info: dict[int, ImageInfo] = {}

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
        return np.array(
            Image.open(
                os.path.join(self.data_path, self.image_id2image_name[image_id])
            )
        )

    def get_annotation(self, annotation_id: int):
        return self.annotation_id2annotation[annotation_id]

    def check_missing_images(self) -> list[int]:
        missing_images_ids = []
        for image_id, image_name in self.image_id2image_name.items():
            if not os.path.exists(os.path.join(self.data_path, image_name)):
                missing_images_ids.append(image_id)

        return missing_images_ids
    
    def check_annotations_without_image(self) -> list[int]:
        annotations_without_image = []
        for annotation_id, image_id in self.annotation_id2image_id.items():
            if image_id not in self.image_id2image_name:
                annotations_without_image.append(annotation_id)

        return annotations_without_image

    def check_not_used_images(self):
        not_used_images_ids = []
        for image_id, annotations_ids in self.image_id2annotation_ids.items():
            if len(annotations_ids) == 0:
                not_used_images_ids.append(image_id)

        return not_used_images_ids
    
    def remove_image(self, image_id: int):
        image_name = self.image_id2image_name[image_id]
        os.remove(os.path.join(self.data_path, image_name))

        self.image_id2image_name.pop(image_id)
        self.image_id2image_dimensions.pop(image_id)

        for annotation_id in self.image_id2annotation_ids[image_id]:
            self.annotation_id2annotation.pop(annotation_id)
            self.annotation_id2image_id.pop(annotation_id)

        self.image_id2annotation_ids.pop(image_id)

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

    def crop_single_image(self, image_id: str, left: int, top: int, right: int, bottom: int):
        """
        TODO: Add documentation
        """
        image = Image.open(os.path.join(self.data_path, self.image_id2image_name[image_id]))
        image = image.crop((left, top, right, bottom))
        image.save(os.path.join(self.data_path, self.image_id2image_name[image_id]))

        self.image_id2image_dimensions[image_id] = (bottom - top, right - left)

        for annotation_id in self.image_id2annotation_ids[image_id]:
            annotation = self.annotation_id2annotation[annotation_id]

            annotation.x -= left
            annotation.y -= top

            if isinstance(annotation, SegmentationAnnotation):
                for point in annotation.points:
                    point.x -= left
                    point.y -= top

    def crop_multiple_images(self, left: int, top: int, right: int, bottom: int):
        """
        TODO: Add documentation
        """
        for image_id in self.image_id2image_name.keys():
            self.crop_single_image(image_id, left, top, right, bottom)

    def crop_images_by_annotations(self, margin: int):
        """
        TODO: Add documentation
        """
        for image_id in self.image_id2image_name.keys():
            x_values = []
            y_values = []
            for annotation_id in self.image_id2annotation_ids[image_id]:
                annotation = self.annotation_id2annotation[annotation_id]
                x_values.append(annotation.x)
                x_values.append(annotation.x + annotation.width)
                y_values.append(annotation.y)
                y_values.append(annotation.y + annotation.height)

            if len(x_values) == 0 or len(y_values) == 0:
                continue
            
            x_min = min(x_values) - margin
            x_min = max(x_min, 0)

            x_max = max(x_values) + margin
            x_max = min(x_max, self.image_id2image_dimensions[image_id][0])

            y_min = min(y_values) - margin
            y_min = max(y_min, 0)

            y_max = max(y_values) + margin
            y_max = min(y_max, self.image_id2image_dimensions[image_id][1])

            self.crop_single_image(image_id, x_min, y_min, x_max, y_max)    