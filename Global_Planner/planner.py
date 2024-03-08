import cv2 as cv
import numpy as np
import math


map = cv.imread("IITKGP Map.png")
map_gray = cv.cvtColor(map, cv.COLOR_BGR2GRAY)
map_gray = cv.resize(map_gray, (1600,800))
sample_map = np.zeros((1600,800), dtype=np.uint8)




cv.imshow("map", map_gray)
k = cv.waitKey(0) & 0xFFF
cv.destroyAllWindows()