import utils

def align_shape(im1, im2):
  '''
  im1: input edge image 1
  im2: input edge image 2

  Output: transformation T [3] x [3]
  '''
  
  # im1 = convertBGRTOGRAY(im1)
  # im2 = convertBGRTOGRAY(im2)
  x1,y1 = np.nonzero(im1)
  x2,y2 = np.nonzero(im2)

  
  scale_X, scale_Y, T_X, T_Y = initializer(x1,y1,x2,y2)
  m1 = 1 #scale_X
  m2 = 0
  m3 = 0
  m4 = 1 #scale_Y
  # print("Image 1 shape : ",im1.shape)
  # print("Image 2 shape : ",im2.shape)

  for i in range(len(x1)):
    x1[i] = m1 * x1[i] + m2 * y1[i] + T_X
    y1[i] = m3 * x1[i] + m4 * y1[i] + T_Y

  for i in range(0):
   
    # print(" Iteration : ",i)
    matches = nearestNeighbourMatches(x1,y1,x2,y2)
    mat_a = create_A_matrix(x1,y1)

    mat_b = create_B_Matrix(matches)

    X = np.matmul(np.linalg.pinv(mat_a),mat_b)

    m1,m2,m3,m4,t1,t2 = X[0],X[1],X[2],X[3],X[4],X[5]
    for i in range(len(x1)):
      x1[i] = m1 * x1[i] + m2 * y1[i]
      y1[i] = m3 * x1[i] + m4 * y1[i]
     
  

  return x1,y1  