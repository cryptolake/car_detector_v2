#!/usr/bin/env python3
"""Draw car bounding box and info on pic."""

from PIL import ImageDraw, ImageFont

def draw_bbox(image, cars, scale=25, color='red', width=3):
    img = ImageDraw.Draw(image)
    for car in cars:
        x0, x1, y0, y1 = car.box
        img.rectangle(((x0, y0), (x1, y1)), outline=color, width=width)
        # top prediction for the car
        cr = car.predictions[0]
        img.text((x0, y0), text=cr.__str__(), 
                 font=ImageFont.truetype("arial.ttf", size=int((x1-x0)//scale)))
    return image

