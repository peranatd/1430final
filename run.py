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

dirToView = ["./P_1/1491423217564_2_-study-dot_test_instructions_frames/"]
sample_train_directory = "./P_1/1491423217564_2_-study-dot_test_instructions_frames/"
sample_test_directory = "./P_1/1491423217564_22_-study-educational_advantages_of_social_networking_sites_instructions_frames/"
dataFile = "gazePredictions.csv"
patchXSize = 40
patchYSize = 15

class Gaze(RNGDataFlow):

  def __init__(self, dirs, dataFile, meta_dir=None, shuffle=None, dir_structure=None):
    # self.imglist = []
    self.imgs = []
    self.labels = []
    self.generate_data(dirs, dataFile)

  def generate_data(self, directory, dataFile):
    with open( directory + dataFile ) as f:
      readCSV = csv.reader(f, delimiter=',')
      for row in readCSV:
        frameFilename = row[0]
        frameTimestamp = row[1]
        # Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
        tobiiLeftEyeGazeX = float( row[2] )
        tobiiLeftEyeGazeY = float( row[3] )
        tobiiRightEyeGazeX = float( row[4] )
        tobiiRightEyeGazeY = float( row[5] )
        webgazerX = float( row[6] )
        webgazerY = float( row[7] )
        clmTracker = row[8:len(row)-1]
        clmTracker = [float(i) for i in clmTracker]
        clmTrackerInt = [int(i) for i in clmTracker]

        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

        img = cv2.imread( frameFilename )

        # Middle of left eye
        for i in range(54,56,2):
          x = clmTrackerInt[i]
          y = clmTrackerInt[i+1]
          leftEye = img[y-(patchYSize-1)/2:y+(patchYSize-1)/2, x-(patchXSize/2):x+(patchXSize/2)-1]
          # img = cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 )

        # Middle of right eye
        for i in range(64,66,2):
          x = clmTrackerInt[i]
          y = clmTrackerInt[i+1]
          rightEye = img[y-(patchYSize-1)/2:y+(patchYSize-1)/2, x-(patchXSize/2):x+(patchXSize/2)-1]
          # img = cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 )

        combined = np.concatenate((leftEye, rightEye), axis=1)
        combined = cv2.resize(combined, (80, 15))
        self.imgs.append(combined)
        # self.labels.append(np.array([tobiiEyeGazeX, tobiiEyeGazeY]))
        self.labels.append(tobiiEyeGazeX)


  def get_data(self):
    for k in np.arange(len(self.imgs)):
      yield [self.imgs[k], self.labels[k]]

  def size(self):
    return len(self.imgs)

def load_data(directory, dataFile):
  data = Gaze(directory, dataFile)
  # augmentators = []
  # data = AugmentImageComponent(data, augmentators)
  data = BatchData(data, 1)
  return data

def main():
  dataset_train = load_data(sample_train_directory, dataFile)
  dataset_test = load_data(sample_test_directory, dataFile)
  config = TrainConfig(
    model = WebGazerModel(),
    dataflow = dataset_train,
    callbacks = [
      InferenceRunner(dataset_test,
        [ScalarStats('cost'), ClassificationError()]
      )
    ],
    max_epoch = 100
    # nr_tower = max(get_nr_gpu(), 1)
  )
  SimpleTrainer(config).train()

main()
