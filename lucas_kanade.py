import utils
def getNextPoints(x, y, im1, im2, ws):
  '''
  Iterative Lucas-Kanade feature tracking
  x,  y : initialized keypoint position in im2
  ws: patch window size

  output: tracked keypoint positions in im2
  '''
  
  G_filter_1 =  gaussian_2D_filter(5,2)
  gaussian_filter = G_filter_1*G_filter_1.T
  #Smoothening the images
  im1 = cv2.filter2D(im1, -1, gaussian_filter)
  im2 = cv2.filter2D(im2, -1, gaussian_filter)
  #X derivative and Y derivative of the image
  Sobel_X = np.array([[-1, 0, 1], 
                      [-2, 0, 2], 
                      [-1, 0, 1]], np.float32)
  
  Sobel_Y = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]], np.float32)
  #Gradients of image 1
  I_x_1 = cv2.filter2D(im1,-1,Sobel_X)
  I_y_1 = cv2.filter2D(im1,-1,Sobel_Y)
  I_xx_1 = cv2.filter2D(np.square(I_x_1), -1, gaussian_filter)
  I_yy_1 = cv2.filter2D(np.square(I_y_1), -1, gaussian_filter)
  I_xy_1 = cv2.filter2D(np.multiply(I_x_1,I_y_1), -1, gaussian_filter)
  #Gradients of Image 2
  I_x_2 = cv2.filter2D(im2,-1,Sobel_X)
  I_y_2 = cv2.filter2D(im2,-1,Sobel_Y)
  #Initilaize x2,y2
  x2 = x
  y2 = y
  #window size
  offset = ws
  #u,v values
  u = np.zeros(im1.shape)
  v = np.zeros(im1.shape)
  keypoints_2 = []
  lost_frames=[]
  # print(cv2.getRectSubPix(I_xx_1,(offset,offset),(np.float32(y[0]),np.float32(x[0]))))
  # print(" Keypoints : ",x2," ",y2)
  for m in range(8):
    for i in range(len(x)):
      if(x2[i] > offset and x2[i]<(im1.shape[0] - offset) and y2[i] > offset and y2[i] < (im1.shape[1] - offset)):

        w_xx = np.sum(cv2.getRectSubPix(I_xx_1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0]))))
        w_yy = np.sum(cv2.getRectSubPix(I_yy_1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0]))))
        w_xy = np.sum(cv2.getRectSubPix(I_xy_1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0]))))
        I_t = cv2.getRectSubPix(im2,(offset,offset),(np.float32(x2[0]),np.float32(y2[0]))) - cv2.getRectSubPix(im1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0])))
        I_xt = np.sum((cv2.getRectSubPix(I_x_1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0])))) * I_t)
        I_yt = np.sum((cv2.getRectSubPix(I_x_1,(offset,offset),(np.float32(x2[0]),np.float32(y2[0])))) * I_t)

        A_T_A = np.array([[w_xx, w_xy], 
                          [w_xy, w_yy]], np.float32)
        A_T_B = np.array([[-I_xt], 
                          [-I_yt]], np.float32)
        sol = np.matmul(np.linalg.pinv(A_T_A), A_T_B)
        # u[x, y], v[x, y] = sol

        # print(" Solution u,v : ",sol)
        x2[i] += sol[0]
        y2[i] += sol[1]
      else:
        lost_frames.append(np.array([x2[i],y2[i]]))
  
  return x2,y2,lost_frames


def trackPoints(pt_x, pt_y, im, ws):
  '''
  Tracking initial points (pt_x, pt_y) across the image sequence
  Outputs:
    track_x: [Number of keypoints] x [2]
    track_y: [Number of keypoints] x [2]
  '''
  lost_frames_points = []
  N = np.prod(pt_x.shape)
  nim = len(im)
  track_x = np.zeros((N, nim))
  track_y = np.zeros((N, nim))
  track_x[:,0] = pt_x
  track_y[:,0] = pt_y
  lost_frames = []
  for t in range(nim-1):
    track_x[:, t+1], track_y[:, t+1],lost_frames= getNextPoints(track_x[:, t], track_y[:, t], im[t,:,:], im[t+1,:,:], ws)
  for i in range(len(lost_frames)):
    lost_frames_points.append(lost_frames[i])
  return track_x, track_y,np.array(lost_frames_points)