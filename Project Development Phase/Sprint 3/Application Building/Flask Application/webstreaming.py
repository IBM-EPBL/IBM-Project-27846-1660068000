from skimage.transform import resize
import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask, render_template, Response
import tensorflow as tf
from gtts import gTTS
global graph
global writer

graph = tf.get_default_graph()
writer = None

model = load_model('model.h5 ')

vals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

app = Flask(_name_)

print("[INFO] accessing video stream... ")
vs = cv2.VideoCapture(0)  # triggers the local camera

pred = ""


@app.route('/')
def index():
    return render_template('index. html ')
