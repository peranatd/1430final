from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf

class WebGazerModel(ModelDesc):

  def __init__(self):
    super(WebGazerModel, self).__init__()
    self.use_bias = True

  def _get_inputs(self):
    return [InputDesc(tf.float32, [None, 15, 40, 3], 'input'),
      InputDesc(tf.float32, [None], 'labelX'),
      InputDesc(tf.float32, [None], 'labelY')
    ]

  def _build_graph(self, inputs):
    image, labelX, labelY = inputs

    #####################################################################
    # TASK 1: Change architecture (to try to improve performance)
    # TASK 1: Add dropout regularization

    # Declare convolutional layers
    #
    # TensorPack: Convolutional layer
    # 10 filters (out_channel), 9x9 (kernel_shape),
    # no padding, stride 1 (default settings)
    # with ReLU non-linear activation function.
    # logits = Conv2D('conv0', image, 10, (9,9), padding='valid', stride=(1,1), nl=tf.nn.relu)
    #
    # TensorPack: Max pooling layer
    # Chain layers together using reference 'logits'
    # 7x7 max pool, stride = none (defaults to same as shape), padding = valid
    # logits = MaxPooling('pool0', logits, (7,7), stride=None, padding='valid')
    #
    # TensorPack: Fully connected layer
    # number of outputs = number of categories (the 15 scenes in our case)
    # logits = FullyConnected('fc0', logits, hp.category_num, nl=tf.identity)
    #####################################################################

    logits = Conv2D('conv1', image, 32, (3,3), nl=tf.nn.relu)
    logits = Conv2D('conv2', logits, 32, (3,3), nl=tf.nn.relu)
    logits = Conv2D('conv3', logits, 64, (3,3), nl=tf.nn.relu)
    logits = MaxPooling('pool1', logits, (3,3), stride=(2,2), padding='valid')
    logits = Conv2D('conv4', logits, 80, (3,3), nl=tf.nn.relu)
    logits = Conv2D('conv5', logits, 192, (3,3), nl=tf.nn.relu)
    logits = MaxPooling('pool2', logits, (2,2), stride=(2,2), padding='valid')
    # logits = Dropout(logits, keep_prob=0.5)

    logitsX = FullyConnected('fc0_x', logits, 9600, nl=tf.nn.relu)
    logitsX = Dropout(logitsX, keep_prob=0.7)
    logitsX = FullyConnected('fc1_x', logitsX, 1000, nl=tf.nn.relu)
    logitsX = Dropout(logitsX, keep_prob=0.7)
    logitsX = FullyConnected('fc2_x', logitsX, 50, nl=tf.identity)

    logitsY = FullyConnected('fc0_y', logits, 9600, nl=tf.nn.relu)
    logitsY = Dropout(logitsY, keep_prob=0.7)
    logitsY = FullyConnected('fc1_y', logitsY, 1000, nl=tf.nn.relu)
    logitsY = Dropout(logitsY, keep_prob=0.7)
    logitsY = FullyConnected('fc2_y', logitsY, 50, nl=tf.identity)

    logitsX = tf.reduce_sum(logitsX, 1)
    logitsY = tf.reduce_sum(logitsY, 1)

    logitsX = tf.identity(logitsX, name='predictedX')
    logitsY = tf.identity(logitsY, name='predictedY')

    # logitsX = tf.Print(logitsX, ["PredictedX", logitsX, tf.shape(logitsX)])
    # labelX = tf.Print(labelX, ["LabelsX", labelX, tf.shape(labelX)])
    # logitsY = tf.Print(logitsY, ["PredictedY", logitsY, tf.shape(logitsY)])
    # labelY = tf.Print(labelY, ["LabelsY", labelY, tf.shape(labelY)])


    # cost = tf.sqrt(tf.reduce_mean(tf.add(tf.squared_difference(labelX, logitsX), tf.squared_difference(labelY, logitsY))))
    costX = tf.reduce_mean(tf.squared_difference(labelX, logitsX))
    costY = tf.reduce_mean(tf.squared_difference(labelY, logitsY))
    cost = tf.sqrt(costX + costY)

    # monitor training error
    add_moving_summary(tf.reduce_mean(cost, name='train_error'))


    #####################################################################
    # TASK 1: If you like, you can add other kinds of regularization,
    # e.g., weight penalization, or weight decay



    #####################################################################


    # Set costs and monitor them for TensorBoard
    # add_moving_summary(cost)
    add_param_summary(('.*/kernel', ['histogram']))   # monitor W
    self.cost = tf.add_n([tf.reduce_mean(cost)], name='error')


  def _get_optimizer(self):
    lr = get_scalar_var('learning_rate', 1e-4, summary=True)

    # Use gradient descent as our optimizer
    opt = tf.train.GradientDescentOptimizer(lr)

    # There are many other optimizers - https://www.tensorflow.org/api_guides/python/train#Optimizers
    # Including the momentum-based gradient descent discussed in class.

    return opt
