import cv2
import numpy as np
import os
from numpy import linalg as LA
import math
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy import ndimage,signal
import matplotlib.pyplot as plt
import time
from operator import ne
from scipy.io.netcdf import IS_PYPY



def readImages(folder, num_images):
  arr_images = []
  for i in range(num_images):
    arr_images.append(cv2.imread(f'{folder}hotel.seq{i}.png'))
  return np.array(arr_images, dtype=np.float32)
def convertRGBTOGRAY(image):
  
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  return image
def convertBGRTORGB(image):
  
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image 
def convertBGRTOGRAY(image):
  
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image 
def gaussian_2D_filter(ksize, cutoff_frequency):
 
  filter = cv2.getGaussianKernel(ksize,cutoff_frequency)
  return filter
def imgfilter(image, filter):
  output_img = cv2.sepFilter2D(image,-1,filter,filter)
  return output_img
def convertTouint8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)


def calDistance(x1,y1,x2,y2):
  return np.sqrt(np.sum([np.square(x2-x1),np.square(y2-y1)]))

def evalAlignment(aligned1, im2):
  '''
  Computes the error of the aligned image (aligned1) and im2, as the
  average of the average minimum distance of a point in aligned1 to a point in im2
  and the average minimum distance of a point in im2 to aligned1.
  '''
  d2 = ndimage.distance_transform_edt(1-im2) #distance transform
  err1 = np.mean(np.mean(d2[aligned1 > 0]))
  d1 = ndimage.distance_transform_edt(1-aligned1);
  err2 = np.mean(np.mean(d2[im2 > 0]))
  err = (err1+err2)/2;
  return err

def displayAlignment(im1, im2, aligned1, thick=False):
  '''
  Displays the alignment of im1 to im2
     im1: first input image to alignment algorithm (im1(y, x)=1 if (y, x) 
      is an original point in the first image)
     im2: second input image to alignment algorithm
     aligned1: new1(y, x) = 1 iff (y, x) is a rounded transformed point from the first time 
     thick: true if a line should be thickened for display
  ''' 
  if thick:
    # for thick lines (looks better for final display)
    dispim = np.concatenate((cv2.dilate(im1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(aligned1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(im2.astype('uint8'), np.ones((3,3), np.uint8), iterations=1)), axis=-1)
  else:
    # for thin lines (faster)
    dispim = np.concatenate((im1, aligned1, im2), axis = 1)
  return dispim

def create_A_matrix(x,y):
  A = np.empty((2*x.shape[0],6))
  for i in range(A.shape[0]):
    j = i//2
    if i%2 == 0:
      A[i,0] = x[j]
      A[i,1] = y[j]
      A[i,2] , A[i,3], A[i,5]= 0,0,0
      A[i,4] = 1
    else:
      A[i,2] = x[j]
      A[i,3] = y[j]
      A[i,0] , A[i,1], A[i,4]= 0,0,0
      A[i,5] = 1


  return A

def create_B_Matrix(matches):
  size = 2 * len(matches)
  b = np.empty((size,1))
  for i in range(b.shape[0]):
    j = i//2
    if i%2 == 0:
      b[i] = matches[j][0]
    else:
      b[i] = matches[j][1]
  return b



def nearestNeighbourMatches(x1,y1,x2,y2):
  matches = []
  args = []
  for i in range(len(x1)):
    for j in range(len(x2)):
      dist = calDistance(x1[i],y1[i],x2[j],y2[j])
      args.append(dist)
    minimum = min(args)
    index = args.index(minimum)
    matches.append((x2[index],y2[index]))
    args.clear()
  return matches

def initializer(x1,y1,x2,y2):
  mean_x1 = np.mean(x1)
  mean_y1 = np.mean(y1)
  mean_x2 = np.mean(x2)
  mean_y2 = np.mean(y2)
  var_x1 = np.var(x1)
  var_y1 = np.var(y1)
  var_x2 = np.var(x2)
  var_y2 = np.var(y2)

  scale_X = var_x1/var_x2
  scale_Y = var_y1/var_y2

  T_X = mean_x1 - mean_x2
  T_Y = mean_y1 - mean_y2

  return scale_X, scale_Y, T_X, T_Y

def calculateSSD(img1, img2):
  # print(puzzle.shape, "  ",img.shape )
  if img1.shape != img2.shape:
    print(img1.shape, "  ",img1.shape )
    print(img2.shape, "  ",img2.shape )
    print(" Not same dimensions ")
    return -1
  else:
    X = np.subtract(img1,img2)
    ssd = np.sum(np.square(X))
    return ssd
def matcherMethod(arr):
  max = np.max(arr)
  min1 = np.min(arr)
  min1_index = arr.index(min1)

  arr[min1_index] = max

  min2 = np.min(arr)
  return min1 ,min2, min1_index

def meanDistance(arr):
  sum = 0
  for m in arr:
    sum += m[2]

  # print(sum/arr.shape[0])
  return sum/arr.shape[0]