import os
import csv
import cv2
import numpy as np

test_or_train = "train"
patchXSize = 40
patchYSize = 15

with open( test_or_train + "_1430_1.txt" ) as dat:
  content = dat.readlines()
  content = [x.strip() for x in content] 
  c1 = 0

  for dirToView in content:
    c1 = c1+1 # count folders
    dirToView = dirToView.replace("\\", "/") + '/'
    dataFile = "gazePredictions.csv"
    if not os.path.exists( dirToView + dataFile ): continue

    with open( dirToView + dataFile ) as f:
      readCSV = csv.reader(f, delimiter=',')
      c2 = 0

      for row in readCSV:
        c2 = c2+1 # count frames
        frameFilename = row[0]
        clmTracker = row[8:len(row)-1]
        clmTracker = [float(i) for i in clmTracker]
        clmTrackerInt = [int(i) for i in clmTracker]
        if not os.path.exists( frameFilename ): continue
        img = cv2.imread( frameFilename )

        # Middle of left eye
        for i in range(54,56,2):
          x = clmTrackerInt[i]
          y = clmTrackerInt[i+1]
          leftEye = img[y-(patchYSize-1)/2:y+(patchYSize-1)/2, x-(patchXSize/2):x+(patchXSize/2)-1]

        # Middle of right eye
        for i in range(64,66,2):
          x = clmTrackerInt[i]
          y = clmTrackerInt[i+1]
          rightEye = img[y-(patchYSize-1)/2:y+(patchYSize-1)/2, x-(patchXSize/2):x+(patchXSize/2)-1]

        height, width, channel = leftEye.shape
        if height == 0 or width == 0: continue
        height, width, channel = rightEye.shape
        if height == 0 or width == 0: continue

        leftEye = cv2.resize(leftEye, (patchXSize, patchYSize))
        rightEye = cv2.resize(rightEye, (patchXSize, patchYSize))

        offsetX = 8
        offsetY = 2
        leftEye = cv2.cvtColor(leftEye,cv2.COLOR_BGR2GRAY)
        l_ori = leftEye
        leftEye = cv2.equalizeHist(leftEye)
        retval, leftEye = cv2.threshold(leftEye, 20, 255, cv2.THRESH_BINARY)
        l_circles = cv2.HoughCircles(leftEye[offsetY:patchYSize-offsetY, offsetX:patchXSize-offsetX],cv2.HOUGH_GRADIENT,1,100,param1=5,param2=5,minRadius=4,maxRadius=8)

        rightEye = cv2.cvtColor(rightEye,cv2.COLOR_BGR2GRAY)
        r_ori = rightEye
        rightEye = cv2.equalizeHist(rightEye)
        retval, rightEye = cv2.threshold(rightEye, 20, 255, cv2.THRESH_BINARY)
        r_circles = cv2.HoughCircles(rightEye[offsetY:patchYSize-offsetY, offsetX:patchXSize-offsetX],cv2.HOUGH_GRADIENT,1,100,param1=5,param2=5,minRadius=4,maxRadius=8)
        if l_circles is not None and r_circles is not None :
          cv2.imwrite('./' + test_or_train + '_leftEye_filtered/' + str(c1) + '_' + str(c2) + '.png', l_ori)
          cv2.imwrite('./' + test_or_train + '_rightEye_filtered/' + str(c1) + '_' + str(c2) + '.png', r_ori)
