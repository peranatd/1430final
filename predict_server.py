##### RUN THIS BLOCK TO SAVE MODEL

# from tensorpack import *
# from webgazermodel import WebGazerModel
# from tensorpack.tfutils.export import ModelExport

# e = ModelExport(WebGazerModel(), ['input'], ['predictedX', 'predictedY'])
# e.export('train_log/run/checkpoint', 'export')

##############################

# RUN BELOW TO 'PREDICT'

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.saved_model import tag_constants

eyepatch = cv2.imread('eyepatches/test_leftEye_filtered/1_1.png', 0)
normed = np.zeros((15, 40, 3), 'float32')
eyepatch = (eyepatch - np.mean(eyepatch)) / np.std(eyepatch)
normed[..., 0] = eyepatch
normed[..., 1] = eyepatch
normed[..., 2] = eyepatch
normed = np.array([normed, normed, normed])

export_dir = "export/"
with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)

    prediction = tf.get_default_graph().get_tensor_by_name('predictedX:0')
    img = tf.get_default_graph().get_tensor_by_name('input:0')

    prediction = sess.run(prediction, {img: normed})
    print(prediction)
