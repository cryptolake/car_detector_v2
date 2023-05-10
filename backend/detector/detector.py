#!/usr/bin/python3
"""Api to car model detection."""
from . import brand_model, color, extraction_model, image_processor
from PIL import Image
import torch
import numpy as np

class Predict:
    """
    Class to predict car(s) model in image.

    This class acts as an api to the model, it is initialzed
    by the image to detect and it returns results and stats.
    """

    def __init__(self, image, threshold, top):
        """
        Initialize the class.

        image: image to initialize with
        """
        self.image = Image.open(image).convert('RGB')
        self.array_image = np.array(self.image)
        self.cars = self.extract_cars(threshold)
        self.prediction = self.predict(top)
        

    def extract_cars(self, threshold):
        """
        Returns the (x, y, w, h) of each bounding box of each car
        """
        cars = []
        inputs = image_processor(images=self.array_image, return_tensors="pt")
        outputs = extraction_model(**inputs)
        target_sizes = torch.tensor([self.image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs,
                                                                threshold=threshold,
                                                                target_sizes=target_sizes)[0]
        for _, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if extraction_model.config.id2label[label.item()] in ['car', 'truck']:
                x0, y0, x1, y1 = np.round(box.tolist()).astype(np.int64)
                if (x1-x0) > 50 and (y1-y0) > 50:
                    car = self.array_image[y0:y1, x0:x1]
                    cars.append((car, (x0, y0, x1, y1)))
        return cars


    def predict(self, top):
        predictions = []
        for car, box in self.cars:
            pred = brand_model.predict(car)
            top_pred = sorted(list(map(lambda x: (brand_model.dls.vocab[x[0]],
                                       float(x[1])),enumerate(pred[2]))),
                   reverse=True, key=lambda x: x[1])[:top]
            pred_color = color.predict(car)
            car_predictions = ([], box)
            for pred, prob in top_pred:
                pred = pred.split()
                year, brand = pred[:2]
                model = " ".join(pred[2:])
                prediction = {"Year": year,
                              "Brand": brand,
                              "Model": model,
                              "Color": pred_color[0],
                              "Prob": prob}
                car_predictions[0].append(prediction)
            predictions.append(car_predictions)
        return predictions
