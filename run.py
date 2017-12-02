import os
import csv
import cv2

dirToView = "./P_1/1491423217564_2_-study-dot_test_instructions_frames/"
dataFile = "gazePredictions.csv"
patchXSize = 40
patchYSize = 15

with open( dirToView + dataFile ) as f:
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

    cv2.imshow( 'viewer.py', rightEye )
    cv2.waitKey(0)
