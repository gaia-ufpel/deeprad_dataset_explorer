import json
import datetime

import cv2
from pycocotools import mask as mask_utils

from dataset import Dataset, Point, SegmentationAnnotation

class COCOAdapter:
    @staticmethod
    def load(json_path: str, images_path: str) -> Dataset:
        """
        TODO: Add documentation
        """
        coco_dataset = None
        with open(json_path, 'r') as reader:
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

            points = []
            if type(annotation_info['segmentation']) == dict:
                pyObj = mask_utils.frPyObjects(
                    annotation_info["segmentation"],
                    annotation_info["segmentation"]["size"][0],
                    annotation_info["segmentation"]["size"][1],
                )
                maskedArr = mask_utils.decode(pyObj)
                contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    for point in contour:
                        x, y = point[0]
                        x = float(x)
                        y = float(y)
                        points.append(Point(x, y))
            else:
                for i in range(0, len(annotation_info['segmentation'][0]), 2):
                    x = annotation_info['segmentation'][0][i]
                    y = annotation_info['segmentation'][0][i+1]
                    points.append(Point(x, y))

            assert len(points) > 3, "The segmentation must have at least three points!"

            annotation = SegmentationAnnotation(
                class_id = annotation_info['category_id'],
                x = annotation_info['bbox'][0],
                y = annotation_info['bbox'][1],
                width = annotation_info['bbox'][2],
                height = annotation_info['bbox'][3],
                area = annotation_info['area'],
                points = points
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
    def _get_coco_template():
        return {
            "info":{
                "description":"",
                "url":"",
                "version":"",
                "year":datetime.datetime.now().year,
                "date_created":datetime.datetime.now().strftime("%Y-%m-%d")
            },
            "licenses":[
                {
                    "url":"",
                    "id":0,
                    "name":""
                }
            ],
            "images":[
                
            ],
            "annotations":[
                
            ],
            "categories":[
                
            ]
        }

    @staticmethod
    def save(dataset: Dataset, json_path: str) -> None:
        """
        TODO: Add documentation
        """
        coco_dataset = COCOAdapter._get_coco_template()
        
        for image_id, image_name in dataset.image_id2image_name.items():
            coco_dataset['images'].append({
                "id": image_id,
                "file_name": image_name,
                "width": dataset.image_id2image_dimensions[image_id][0],
                "height": dataset.image_id2image_dimensions[image_id][1],
                "license": 0
            })

        for annotation_id, annotation in dataset.annotation_id2annotation.items():
            ann = {
                "id": annotation_id,
                "image_id": dataset.annotation_id2image_id[annotation_id],
                "category_id": annotation.class_id,
                "segmentation": [],
                "area": annotation.area,
                "bbox": [annotation.x, annotation.y, annotation.width, annotation.height],
                "iscrowd": 0
            }
            for point in annotation.points:
                ann['segmentation'].extend([point.x, point.y])
            coco_dataset['annotations'].append(ann)

        for class_id, class_name in dataset.id2class.items():
            coco_dataset['categories'].append({
                "id": class_id,
                "name": class_name,
                "supercategory": ""
            })

        with open(json_path, 'w') as writer:
            writer.write(json.dumps(coco_dataset, indent=4))