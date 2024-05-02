import yaml
import json
import os
import math
import glob
from PIL import Image
from logging import Logger
import shutil
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
class SegmentationAnnotation(ClassificationAnnotation):
    points: list[Point]

class Dataset:
    def __init__(self, id2class: dict[int, str], data_path: str, dataset_task: str, image_id2image_path: dict[int, str], image_dimensions: dict[int, tuple[int, int]], dataset_division: dict[str, list[int]], annotation_id2image_id: dict[int, int], annotation_id2annotations: dict[int, list[Annotation]]):
        self.id2class = id2class
        self.class2id = {v: k for k, v in id2class.items()}
        self.data_path = data_path
        self.dataset_task = dataset_task

        self.image_id2image_path: dict[int, str] = image_id2image_path
        self.image_path2image_id: dict[str, int] = {v: k for k, v in image_id2image_path.items()}
        self.images_dimensions: dict[int, tuple[int, int]] = image_dimensions
        self.dataset_division: dict[str, list[int]] = dataset_division
        self.annotation_id2image_id: dict[int, int] = annotation_id2image_id
        self.image_id2annotation_ids: dict[int, list[int]] = {image_id: [] for image_id in image_id2image_path.keys()}
        for annotation_id, image_id in annotation_id2image_id.items():
            self.image_id2annotation_ids[image_id].append(annotation_id)
        self.annotation_id2annotations: dict[int, list[Annotation]] = annotation_id2annotations

    def split_dataset(self, train_ratio: float, val_ratio: float, test_ratio: float):
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("The sum of the ratios must be 1")

        images = list(self.id2image_path.keys())
        train_size = math.ceil(len(images) * train_ratio)
        val_size = math.ceil(len(images) * val_ratio)

        self.dataset_division['train'] = images[:train_size]
        self.dataset_division['val'] = images[train_size:train_size + val_size]
        self.dataset_division['test'] = images[train_size + val_size:]

    def copy_dataset(self, destination_path: str):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        for image_path in self.id2image_path.values():
            shutil.copy(image_path, destination_path)

        return Dataset(self.id2class, destination_path)

    def crop_images(self, height: int, width: int):
        pass

    def crop_images_by_annotations(self, margin: int):
        for image_id, image_path in self.image_id2image_path.items():
            image = Image.open(image_path)
            annotations_ids = self.image_id2annotation_ids[image_id]

            x_values = []
            y_values = []
            for annotation_id in annotations_ids:
                annotation = self.annotation_id2annotations[annotation_id]

                if isinstance(annotation, SegmentationAnnotation):
                    points = annotation.points
                    x_values.extend([point.x for point in points])
                    y_values.extend([point.y for point in points])


            x_min = min(x_values) - margin
            x_max = max(x_values) + margin
            y_min = min(y_values) - margin
            y_max = max(y_values) + margin
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image.save(image_path)

            for annotation_id in annotations_ids:
                annotation = self.annotation_id2annotations[annotation_id]

                if isinstance(annotation, SegmentationAnnotation):
                    for point in annotation.points:
                        point.x -= x_min
                        point.y -= y_min
            
        