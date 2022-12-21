import utils

def getKeypoints(img, thresh,k,sigma):
  '''
  Detecting keypoints using Harris corner criterion
  img: input image
  thresh: threshold 
  
  output: (N,2) array of [x,y] keypoints
  '''
  # Converting image into grayscale
  colour_img = np.copy(img)
  img = convertBGRTOGRAY(img)
  
  # Gaussian Blur
  G_filter =  gaussian_2D_filter(k,sigma)
  image = imgfilter(img,G_filter)
  

  G_filter_1 =  gaussian_2D_filter(k,sigma)
  gaussian_filter = G_filter_1*G_filter_1.T

  #X derivative and Y derivative of the image
  Sobel_X = np.array([[-1, 0, 1], 
                      [-2, 0, 2], 
                      [-1, 0, 1]], np.float32)
  Sobel_Y = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]], np.float32)

  I_x = cv2.filter2D(image,-1,Sobel_X)
  I_y = cv2.filter2D(image,-1,Sobel_Y)
 


  I_x_2 = cv2.filter2D(np.square(I_x), -1, gaussian_filter)
  I_y_2 = cv2.filter2D(np.square(I_y), -1, gaussian_filter)
  I_xy = cv2.filter2D(np.multiply(I_x, I_y), -1, gaussian_filter)

  max = 0
  offset = 5
  harris_Response_factor = np.zeros(image.shape, dtype=np.float32)

  #method - 1
  harris_Response_factor = (I_x_2 * I_y_2) - (I_xy * I_xy) - 0.05 * ((I_x_2 + I_y_2) **2)

# Method - 2
  # for i  in range(1,image.shape[0] - 1 ):
  #   for j in range(1, image.shape[1] - 1):
  #     S_xx = np.sum(I_x_2[i - int(offset/2) : i + int(offset/2) + 1 , j - int(offset/2): j + int(offset/2) + 1])
  #     S_yy = np.sum(I_y_2[i - int(offset/2) : i + int(offset/2) + 1, j - offset: j + int(offset/2) + 1])
  #     S_xy = np.sum(I_xy[i - int(offset/2) : i + int(offset/2) + 1, j - int(offset/2): j + int(offset/2) + 1])
  #     D = (S_xx * S_yy) - (S_xy * S_xy)
  #     T = (S_xx + S_yy)
  #     harris_Response_factor[i,j] = D - 0.05 * T * T
  #     if harris_Response_factor[i,j] > max :
  #       max = harris_Response_factor[i,j]

  harris_Response_factor = harris_Response_factor / np.max(harris_Response_factor)
  
  # NMS
  window_shape = 2
  corner_image = np.zeros_like(harris_Response_factor)

  for i in range(window_shape, image.shape[0]-window_shape):
    for j in range(window_shape, image.shape[1] -window_shape):
      window = harris_Response_factor[i-window_shape : i + window_shape + 1, j -window_shape : j + window_shape + 1]
      max_response = np.max(window)
      z = np.zeros((5,5))
      m,n = np.where(window == max_response) 
      window = z
      window[m,n] = max_response
      harris_Response_factor[i-window_shape : i + window_shape + 1, j -window_shape : j + window_shape + 1] = window

  count = 0
  corners = []
  for i in range(harris_Response_factor.shape[0]):
    for j in range(harris_Response_factor.shape[1]):
      if harris_Response_factor[i,j]> thresh:
        count+=1
        corners.append(np.array([i,j]))
        cv2.circle(colour_img,(j,i),2,(0,255,0),-1)
  # cv2_imshow(colour_img)
  print("Count of Corners : ", count)


  return np.array(corners), convertTouint8(colour_img)


# #To see the detected in the image
# corners, img = getKeypoints(im[0], thresh,5,1)
# x, y = corners[:,0],corners[:,1]
# End of code