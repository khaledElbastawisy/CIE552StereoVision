# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Visualize the actual 2D points and the projected 2D points calculated
# from the projection matrix
# You do not need to modify anything in this function, although you can if
# you want to.
def evaluate_points(M, Points_2D, Points_3D):

    reshaped_points = np.concatenate(
        (Points_3D, np.ones((Points_3D.shape[0], 1))), axis=1)
    Projection = np.matmul(M, np.transpose(reshaped_points))
    Projection = np.transpose(Projection)
    u = np.divide(Projection[:, 0], Projection[:, 2])
    v = np.divide(Projection[:, 1], Projection[:, 2])
    Residual = np.sum(
        np.power(
            np.power(u - Points_2D[:, 0], 2) +
            np.power(v - Points_2D[:, 1], 2), 0.5))
    Projected_2D_Pts = np.transpose(np.vstack([u, v]))

    return Projected_2D_Pts, Residual


# Visualize the actual 2D points and the projected 2D points calculated
# from the projection matrix
# You do not need to modify anything in this function, although you can if
# you want to.
def visualize_points(Actual_Pts, Project_Pts):
    plt.scatter(Actual_Pts[:, 0], Actual_Pts[:, 1], marker='o')
    plt.scatter(Project_Pts[:, 0], Project_Pts[:, 1], marker='x')
    plt.legend(('Actual Points', 'Projected Points'))
    plt.show()


# Visualize the actual 3D points and the estimated 3D camera center.
# You do not need to modify anything in this function, although you can if
# you want to.
def plot3dview(Points_3D, camera_center1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Points_3D[:, 0],
               Points_3D[:, 1],
               Points_3D[:, 2],
               c='b',
               marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.elev = 31
    ax.azim = -129

    # draw vertical lines connecting each point to Z=0
    min_z = np.min(Points_3D[:, 2])
    for i in range(0, Points_3D.shape[0]):
        ax.plot(np.array([Points_3D[i, 0], Points_3D[i, 0]]),
                np.array([Points_3D[i, 1], Points_3D[i, 1]]),
                np.array([Points_3D[i, 2], min_z]))

    # if camera_center1 exists, plot it
    if 'camera_center1' in locals():
        ax.scatter(camera_center1[0],
                   camera_center1[1],
                   camera_center1[2],
                   s=100,
                   c='r',
                   marker='x')
        ax.plot(np.array([camera_center1[0], camera_center1[0]]),
                np.array([camera_center1[1], camera_center1[1]]),
                np.array([camera_center1[2], min_z]),
                c='r')

    plt.show()
