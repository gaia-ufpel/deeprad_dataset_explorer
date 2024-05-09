import cv2
import numpy as np
from dataset import Dataset

class SegmentationAnnotator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def annotate(self, image_id: int):
        image = self.dataset[image_id]
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.fillPoly(mask, )
        mask = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        return mask