#!/usr/bin/env python3
from detector import Pred
from sys import argv
import requests
from PIL import ImageDraw, ImageFont
from io import BytesIO

def main():
    url = argv[1]
    min_size = int(argv[2])
    threshold = float(argv[3])
    top = int(argv[4])
    image = requests.get(url, stream=True).raw
    prediction = Pred(BytesIO(image.data), min_size, threshold, top)
    img_rec = ImageDraw.Draw(prediction.image)
    cars = prediction.prediction
    print(f"Found {len(cars)} cars.")
    if len(cars) > 0:
        for car in cars:
            box = car.box
            x0, x1, y0, y1 = box
            img_rec.rectangle(((x0, y0), (x1, y1)), outline='Red', width=3)
            cr = car.predictions[0]
            img_rec.text((x0, y0), text=cr.__str__(), 
                         font=ImageFont.truetype("arial.ttf", size=int((x1-x0)//25)))
            print("\n-----------------------------------------------------------------------\n")
            for i, cr in enumerate(car.predictions):
                if i != 0:
                    print('OR')
                print(cr.__str__())
        print("\n-----------------------------------------------------------------------\n")
        prediction.image.show()

    
main()
