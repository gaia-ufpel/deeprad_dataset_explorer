import cv2
import numpy as np
from dataset import Dataset

class LabelAnnotator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def annotate(self, image_id: int):
        image = self.dataset[image_id]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = cv2.putText(mask, 'Label', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        mask = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        return mask