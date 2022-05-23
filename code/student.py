# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np


# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    # [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the 'np.linalg.lstsq' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD via np.linalg.svd, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    mT1 = []
    mT2 = []
    numPoints = Points_3D.shape[0]
    p3d = Points_3D
    p2d = Points_2D
    for p in range (0, numPoints):
        mT1.append(p3d[p, 0:3].tolist() + [1, 0, 0, 0, 0] + [x * (-p2d[p, 0]) for x in p3d[p, 0:3].tolist()])
        mT1.append([0, 0, 0, 0] + p3d[p, 0:3].tolist() + [1] + [x * (-p2d[p, 1]) for x in p3d[p, 0:3].tolist()])
        mT2.append([p2d[p,0]])
        mT2.append([p2d[p,1]])
    M = np.linalg.lstsq(mT1, mT2)[0]
    M = np.append(M, 1)
    M = np.reshape(M, (3, 4))

    return M


# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################

    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    Center = np.matmul(-np.linalg.inv(M[0:3, 0:3]), M[:, 3])

    return Center
