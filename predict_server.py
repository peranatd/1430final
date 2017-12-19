##### RUN THIS BLOCK TO SAVE MODEL

# from tensorpack import *
# from webgazermodel import WebGazerModel
# from tensorpack.tfutils.export import ModelExport
#
# e = ModelExport(WebGazerModel(), ['input'], ['predictedX', 'predictedY'])
# e.export('train_log/run/checkpoint', 'export')

##############################

# RUN BELOW TO 'PREDICT'

import tensorflow as tf
import numpy as np
import cv2
import io
from tensorflow.python.saved_model import tag_constants
from flask import Flask, request
import json

right_dir = "rightExport/"
left_dir = "leftExport/"
leftGraph = tf.Graph()
rightGraph = tf.Graph()
with tf.Session(graph=rightGraph, config=tf.ConfigProto(allow_soft_placement=True)) as sessRight:
    with tf.Session(graph=leftGraph, config=tf.ConfigProto(allow_soft_placement=True)) as sessLeft:
        tf.saved_model.loader.load(sessRight, [tag_constants.SERVING], right_dir)
        tf.saved_model.loader.load(sessLeft, [tag_constants.SERVING], left_dir)
        imgRight = rightGraph.get_tensor_by_name('input:0')
        imgLeft = leftGraph.get_tensor_by_name('input:0')

        app = Flask(__name__)
        def get_normed_image(image):
            normed = np.zeros((15, 40, 3), 'float32')
            image = (image - np.mean(image)) / np.std(image)
            normed[..., 0] = image
            normed[..., 1] = image
            normed[..., 2] = image
            return np.array([normed])

        def get_prediction(eyepatches):
            left = eyepatches[0]
            right = eyepatches[1]

            normedLeft = get_normed_image(left)
            normedRight = get_normed_image(right)

            leftX = leftGraph.get_tensor_by_name('predictedX:0')
            leftY = leftGraph.get_tensor_by_name('predictedY:0')
            leftX = sessLeft.run(leftX, {imgLeft: normedLeft})
            leftY = sessLeft.run(leftY, {imgLeft: normedLeft})

            rightX = rightGraph.get_tensor_by_name('predictedX:0')
            rightY = rightGraph.get_tensor_by_name('predictedY:0')
            rightX = sessRight.run(rightX, {imgRight: normedRight})
            rightY = sessRight.run(rightY, {imgRight: normedRight})

            return leftX[0], leftY[0], rightX[0], rightY[0]


        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                left = io.BytesIO()
                right = io.BytesIO()
                request.files.get('leftEye').save(left)
                request.files.get('rightEye').save(right)
                left = np.fromstring(left.getvalue(), dtype=np.uint8)
                right = np.fromstring(right.getvalue(), dtype=np.uint8)
                eyepatch = [cv2.imdecode(left, 0), cv2.imdecode(right, 0)]
                result = get_prediction(eyepatch)
                return str(result)
            except:
                return -1

        app.run()
