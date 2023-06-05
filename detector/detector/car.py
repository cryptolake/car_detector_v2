#!/usr/bin/python3
"""Class to represent car information."""
from pydantic import BaseModel

class Car(BaseModel):
    """Class to represent car information."""
    brand: str
    model: str
    year: int
    color: str
    prob: float

    def __str__(self):
        return f"Brand: {self.brand}\nmodel: {self.model}\nYear: {self.year}\nColor: {self.color}\nProb: {self.prob}"

class CarPrediction(BaseModel):
    """Class to represent car(s) prediction(s)."""
    predictions: list[Car]
    box: tuple[int, int, int, int]
