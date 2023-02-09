from __future__ import annotations
from typing import *
from utils import *
import csv
import json
import base64
import numpy as np
import cv2

class CsvStructure:

    def __init__(self, file_path: str, encoding: str = 'utf_8') -> None:
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            self.data = [data for data in reader]
    
    def get_row(self, index: int) -> list[str | int | float]:
        return self.data[index]
    
    def get_column(self, index: int) -> list[str | int | float]:
        return [data[index] for data in self.data]
    
    def get_rows(self, start: int, end: int) -> list[list[str | int | float]]:
        return self.data[start:end]
    
    def get_columns(self, start: int, end: int) -> list[list[str | int | float]]:
        return [data[start:end] for data in self.data]
    
    def remove_row(self, index: int) -> CsvStructure:
        del self.data[index]
        return self
    
    def remove_column(self, index: int) -> CsvStructure:
        for data in self.data:
            del data[index]
        return self
    
    def remove_rows(self, start: int, end: int) -> CsvStructure:
        del self.data[start:end]
        return self
    
    def remove_columns(self, start: int, end: int) -> CsvStructure:
        for data in self.data:
            del data[start:end]
        return self

    # For decisionTree etc...
    # def as_multiply(self, index: int) -> CsvStructure:
    #     self.data = [[index * (d) for d in data] for data in self.data]
    #     return self

    def as_int(self) -> CsvStructure:
        self.data = [[parse_int(d) for d in data] for data in self.data]
        return self
    
    def as_float(self) -> CsvStructure:
        self.data = [[parse_float(d) for d in data] for data in self.data]
        return self
    
    def as_str(self) -> CsvStructure:
        self.data = [[str(d) for d in data] for data in self.data]
        return self

class HyperSpectralImage:

    def __init__(self, file_path: str, size: tuple[int, int]) -> None:
        self.bands = 151
        self.size = size
        with open(file_path,'rb') as f:
            self.buf = np.fromfile(f, dtype=np.uint16, count=-1).reshape(size[1], 151, size[0])
    
    def extract_img(self, wave_length: int) -> np.ndarray:
        band = int((wave_length - 350) / 5)
        return self.buf[:, band, :].astype(np.float32)

class LabelmePolygonImage:

    def __init__(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            jdict = json.load(f)
            buf = np.frombuffer(base64.b64decode(jdict['imageData']), np.uint8)
            self.img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            self.mask = np.zeros(self.img.shape, dtype=np.uint8)
            for obj in jdict['shapes']:
                points = np.array(obj['points'], dtype=np.int32)
                cv2.fillPoly(self.mask, [points], (255, 255, 255))
            self.img = cv2.bitwise_and(self.img, self.mask)
    
    def get_original_img(self) -> np.ndarray:
        return self.img

    def get_mask_img(self) -> np.ndarray:
        return self.mask