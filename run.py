import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages')

import os
import csv
import cv2

from tensorpack import *
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow
from webgazermodel import WebGazerModel

dirToTrain = [line.strip()+'/' for line in open('./train_1430_1.txt')]
dirToTest = [line.strip()+'/' for line in open('./test_1430_1.txt')]
sample_train_directory = "./P_1/1491423217564_2_-study-dot_test_instructions_frames/"
sample_test_directory = "./P_1/1491423217564_29_-study-where_to_find_morel_mushrooms_writing_frames/"
dataFile = "gazePredictions.csv"
patchXSize = 40
patchYSize = 15

class Gaze(RNGDataFlow):

  def __init__(self, dirs, dataFile, test_or_train, meta_dir=None, shuffle=None, dir_structure=None):
    self.imglist = []
    self.imgs = []
    self.labelsX = []
    self.labelsY = []
    self.c1 = 0
    self.c2 = 0
    for d in dirs:
      self.c1 = self.c1 + 1
      self.generate_labels_and_img_list(d, dataFile, test_or_train)

  def generate_labels_and_img_list(self, directory, dataFile, test_or_train):
    with open( directory + dataFile ) as f:
      readCSV = csv.reader(f, delimiter=',')
      self.c2 = 0
      for row in readCSV:
        self.c2 = self.c2 + 1
        leftEyeDir = './eyepatches/' + test_or_train + '_leftEye_filtered/' + str(self.c1) + '_' + str(self.c2) + '.png'
        if not os.path.exists(leftEyeDir): continue

        rightEyeDir = './eyepatches/' + test_or_train + '_rightEye_filtered/' + str(self.c1) + '_' + str(self.c2) + '.png'
        #frameFilename = row[0]

        frameTimestamp = row[1]
        # Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
        tobiiLeftEyeGazeX = float( row[2] )
        tobiiLeftEyeGazeY = float( row[3] )
        tobiiRightEyeGazeX = float( row[4] )
        tobiiRightEyeGazeY = float( row[5] )
        clmTracker = row[8:len(row)-1]
        clmTracker = [float(i) for i in clmTracker]
        clmTrackerInt = [int(i) for i in clmTracker]
        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

        left_eye_x = clmTrackerInt[54]
        left_eye_y = clmTrackerInt[55]
        right_eye_x = clmTrackerInt[64]
        right_eye_y = clmTrackerInt[65]
        self.imglist.append([leftEyeDir, rightEyeDir, left_eye_x, left_eye_y, right_eye_x, right_eye_y])
        #self.labels.append(np.array([tobiiEyeGazeX, tobiiEyeGazeY]))
        # self.labelsX.append(min(49, max(0, np.floor(tobiiLeftEyeGazeX*50))))
        self.labelsX.append(tobiiLeftEyeGazeX)
        self.labelsY.append(tobiiLeftEyeGazeY)

  def generate_data(self, args):
    #frameFilename, left_eye_x, left_eye_y, right_eye_x, right_eye_y = args
    leftEyeDir, rightEyeDir, left_eye_x, left_eye_y, right_eye_x, right_eye_y = args
    leftEye = cv2.imread(leftEyeDir, 0)
    rightEye = cv2.imread(rightEyeDir, 0)
    leftEye = rightEye

    """
    # Middle of left eye
    l_minY, l_maxY = max(left_eye_y-(patchYSize-1)/2, 0), min(left_eye_y+(patchYSize-1)/2, height)
    l_minX, l_maxX = max(left_eye_x-(patchXSize/2), 0),   min(left_eye_x+(patchXSize/2)-1, width)
    leftEye = img[l_minY:l_maxY, l_minX:l_maxX]

    # Middle of right eye
    r_minY, r_maxY = max(right_eye_y-(patchYSize-1)/2, 0), min(right_eye_y+(patchYSize-1)/2, height)
    r_minX, r_maxX = max(right_eye_y-(patchXSize/2), 0),   min(right_eye_y+(patchXSize/2)-1, width)
    rightEye = img[r_minY:r_maxY, r_minX:r_maxX]
    """

    #leftEye = cv2.resize(leftEye, (40, 15))
    #rightEye = cv2.resize(rightEye, (40, 15))

    # combined = np.concatenate((leftEye, rightEye), axis=1)
    #combined = cv2.resize(combined, (80, 15))

    # normalise
    leftEye = (leftEye - np.mean(leftEye)) / np.std(leftEye)
    normed = np.zeros((15, 40, 3), 'float32')
    normed[..., 0] = leftEye
    normed[..., 1] = leftEye
    normed[..., 2] = leftEye

    return normed


  def get_data(self):
    for k in np.arange(len(self.imglist)):
      img = self.generate_data(self.imglist[k]);
      if img is None: continue
      yield [img, self.labelsX[k], self.labelsY[k]]

  def size(self):
    return len(self.imglist)

def load_data(directory, dataFile, test_or_train):
  data = Gaze(directory, dataFile, test_or_train)
  augmentators = [imgaug.MeanVarianceNormalize()]
  data = AugmentImageComponent(data, augmentators)
  data = BatchData(data, 50)
  return data

def main():
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  dataset_train = load_data(dirToTrain, dataFile, "train")
  dataset_test = load_data(dirToTest, dataFile, "test")

  logger.auto_set_dir()

  config = TrainConfig(
    model = WebGazerModel(),
    dataflow = dataset_train,
    callbacks = [
      ModelSaver(),
      MinSaver('val-error-top1'),
      InferenceRunner(dataset_test,
        [ScalarStats('error')]
      )
    ],
    max_epoch = 100
    # nr_tower = max(get_nr_gpu(), 1)
  )
  QueueInputTrainer(config).train()

main()
