#!/usr/bin/python3
"""Api to car model detection."""

from . import brand_model, color, extraction_model, image_processor
from .car import Car, CarPrediction

import numpy as np
from PIL import Image
import torch


class Predict:
    """
    Class to predict car(s) model in image.

    This class acts as an api to the model, it is initialzed
    by the image to detect and it returns results and stats.
    """

    def __init__(self, image, min_size, threshold, top):
        """
        Initialize the class.

        image: image to initialize with
        min_size: minmum size of car in pixels
        threshold: the threshold to consider the prediction a car
        top: the number of top results to show
        """
        self.image = Image.open(image).convert('RGB')
        self.array_image = np.array(self.image)
        self.boxes = self.extract_cars(threshold, min_size)
        self.prediction = self.predict(top)

    def extract_cars(self, threshold, min_size):
        """
        Returns the (x0, y0, x1, y1) of each bounding box of each car
        """
        boxes = []
        inputs = image_processor(images=self.array_image, return_tensors="pt")
        outputs = extraction_model(**inputs)
        target_sizes = torch.tensor([self.image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs,
                                                                threshold=threshold,
                                                                target_sizes=target_sizes)[0]
        for _, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if extraction_model.config.id2label[label.item()] == 'car':
                x0, y0, x1, y1 = np.round(box.tolist()).astype(np.int64)
                if (x1-x0) > min_size and (y1-y0) > min_size:
                    boxes.append((x0, x1, y0, y1))
        return boxes

    def predict(self, top):
        cars = []
        for box in self.boxes:
            x0, x1, y0, y1 = box
            car = self.array_image[y0:y1, x0:x1]
            pred = brand_model.predict(car)
            top_pred = sorted(list(map(lambda x: (brand_model.dls.vocab[x[0]],
                                       float(x[1])),enumerate(pred[2]))),
                   reverse=True, key=lambda x: x[1])[:top]
            pred_color = color.predict(car)
            car_predictions = CarPrediction(predictions=[], box=box)
            for pred, prob in top_pred:
                pred = pred.split()
                year, brand = pred[:2]
                model = " ".join(pred[2:])
                prediction = Car(brand=brand, model=model, year=year,
                                 color=pred_color[0], prob=prob)
                car_predictions.predictions.append(prediction)
            cars.append(car_predictions)
        return cars
