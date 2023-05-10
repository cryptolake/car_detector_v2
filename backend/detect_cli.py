#!/usr/bin/env python3
from detector import Pred
from sys import argv
import requests
from PIL import ImageDraw, ImageFont

def main():
    url = argv[1]
    image = requests.get(url, stream=True).raw
    prediction = Pred(image, 0.7, 5)
    cars = prediction.prediction
    img_rec = ImageDraw.Draw(prediction.image)
    for car, box in cars:
        x0, y0, x1, y1 = box
        img_rec.rectangle(((x0, y0), (x1, y1)), outline='Red', width=3)
        cr = car[0]
        img_rec.text((x1, y1), text=f"Brand: {cr['Brand']}\nModel: {cr['Model']}\nYear: {cr['Year']}\nColor: {cr['Color']}\nProb: {cr['Prob']}", font=ImageFont.truetype("arial.ttf", size=(x1-x0)//25))
        print("\n-----------------------------------------------------------------------\n")
        for cr in car:
            print(f"Brand: {cr['Brand']}\nModel: {cr['Model']}\nYear: {cr['Year']}\nColor: {cr['Color']}\nProb: {cr['Prob']}")
            print('*************')

    print("\n-----------------------------------------------------------------------\n")
    prediction.image.show()

    
main()
