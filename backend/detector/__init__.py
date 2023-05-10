#!/usr/bin/python3
"""Detector module."""
from fastai.vision.all import load_learner
from os import environ
from transformers import YolosImageProcessor, YolosForObjectDetection
modelpath = environ.get('MODEL')
colorpath = environ.get('COLOR')
if modelpath is None or colorpath is None:
    raise AttributeError("Use os env MODEL and COLOR for models path.")
brand_model = load_learner(modelpath)
color = load_learner(colorpath)
extraction_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
from .detector import Predict as Pred
