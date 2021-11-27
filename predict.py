import datetime
import io
import time
from concurrent import futures
import cv2
import grpc
import numpy as np
import tensorflow as tf
from model_rox import FacialExpressionModel
from mss import mss
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from win32api import GetSystemMetrics

import ImageClassification_pb2 as pb2
import ImageClassification_pb2_grpc as pb2_grpc

model = FacialExpressionModel("model.json", "model_weights.h5")

print("Width =", GetSystemMetrics(0))
print("Height =", GetSystemMetrics(1))

mon = {'left': GetSystemMetrics(0) - 200, 'top': GetSystemMetrics(1) - 200, 'width': 224, 'height': 224}

with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB',
            (screenShot.width, screenShot.height),
            screenShot.rgb,
        )
        img = img.resize((48, 48), Image.ANTIALIAS)
        img = img.convert("L")
        _image = np.asarray(img)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        result = model.predict_rox(_image[np.newaxis, :, :, np.newaxis])
        print(st + " | Prediction: ", result)
        cv2.imshow(result, np.array(img))
        if cv2.waitKey(33) & 0xFF in (
            ord('q'),
            27,
        ):
            break
