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

    logits = Conv2D('conv1', image, 32, (3,3), nl=BNReLU)
    logits = Conv2D('conv2', logits, 32, (3,3), nl=BNReLU)
    logits = Conv2D('conv3', logits, 64, (3,3), nl=BNReLU)
    logits = MaxPooling('pool1', logits, (3,3), stride=(2,2), padding='valid')
    logits = Conv2D('conv4', logits, 80, (3,3), nl=BNReLU)
    logits = Conv2D('conv5', logits, 192, (3,3), nl=BNReLU)
    logits = MaxPooling('pool2', logits, (2,2), stride=(2,2), padding='valid')
    # logits = Dropout(logits, keep_prob=0.5)

    logitsX = FullyConnected('fc0_x', logits, 9600, nl=tf.nn.relu)
    logitsX = Dropout(logitsX, keep_prob=0.7)
    logitsX = FullyConnected('fc1_x', logitsX, 1000, nl=tf.nn.relu)
    logitsX = Dropout(logitsX, keep_prob=0.7)
    logitsX = FullyConnected('fc2_x', logitsX, 50, nl=tf.identity)

    # logitsY = FullyConnected('fc0_y', logits, 9600, nl=tf.nn.relu)
    # # logitsY = Dropout(logitsY, keep_prob=0.5)
    # logitsY = FullyConnected('fc1_y', logitsY, 1000, nl=tf.nn.relu)
    # logitsY = FullyConnected('fc2_y', logitsY, 50, nl=tf.identity)

    # Add a loss function based on our network output (logits) and the ground truth labels
    # cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logitsX, labels=labelX)
    # cost = tf.reduce_mean(cost, name='cross_entropy_loss')
    # cost = tf.reduce_sum(tf.subtract(labelX, logitsX))
    # logitsX = tf.reshape(logitsX, [-1])
    logitsX = tf.reduce_sum(logitsX, 1)
    # logitsX = tf.Print(logitsX, ["PredictedX", logitsX, tf.shape(logitsX)])
    # labelX = tf.Print(labelX, ["LabelsX", labelX, tf.shape(labelX)])
    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(labelX, logitsX)))
    # costY = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logitsY, labels=labelY)
    # costY = tf.reduce_mean(costY, name='cross_entropy_loss')
    # cost = tf.reduce_sum([costX, costY])

    #wrong = prediction_incorrect(np.array([logitsX, logitsY]), np.array([labelX, labelY])
    # wrong = prediction_incorrect(logitsX, labelX)
    # wrong = tf.squared_difference(labelX, logitsX)
    # wrong = tf.sqrt(wrong)
    # wrongY = prediction_incorrect(logitsY, labelY)
    # wrong = tf.reduce_mean([wrongX, wrongY])

    # monitor training error
    add_moving_summary(tf.reduce_mean(cost, name='train_error'))


    #####################################################################
    # TASK 1: If you like, you can add other kinds of regularization,
    # e.g., weight penalization, or weight decay



    #####################################################################


    # Set costs and monitor them for TensorBoard
    add_moving_summary(cost)
    add_param_summary(('.*/kernel', ['histogram']))   # monitor W
    self.cost = tf.add_n([cost], name='cost')


  def _get_optimizer(self):
    lr = get_scalar_var('learning_rate', 1e-5, summary=True)

    # Use gradient descent as our optimizer
    opt = tf.train.GradientDescentOptimizer(lr)

    # There are many other optimizers - https://www.tensorflow.org/api_guides/python/train#Optimizers
    # Including the momentum-based gradient descent discussed in class.

    return opt
