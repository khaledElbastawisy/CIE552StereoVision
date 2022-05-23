# Camera Calibration Stencil Code
# Transferred to python by Eleanor Tursman, based on previous work by Henry Hu,
# Grady Williams, and James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
#
# This script
# (1) Loads 2D and 3D data points and images
# (2) Calculates the projection matrix from those points    (you code this)
# (3) Computes the camera center from the projection matrix (you code this)
# (4) Estimates the fundamental matrix                      (you code this)
# (5) Adds noise to the points if asked                     (you code this)
# (6) Estimates the fundamental matrix using RANSAC         (you code this)
#     and filters away spurious matches                                    
# (7) Visualizes the F Matrix with homography rectification
#
# The relationship between coordinates in the world and coordinates in the
# image defines the camera calibration. See Szeliski 6.2, 6.3 for reference.
#
# 2 pairs of corresponding points files are provided
# Ground truth is provided for pts2d-norm-pic_a and pts3d-norm pair
# You need to report the values calculated from pts2d-pic_b and pts3d
import warnings
import numpy as np
import os
import argparse
from student import (calculate_projection_matrix, compute_camera_center)
from helpers import (evaluate_points, visualize_points, plot3dview)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):
    data_dir = os.path.dirname(__file__)
    data_dir = data_dir.replace("code", "data/")
    print(data_dir)
    ########## Part (1)
    Points_2D = np.loadtxt(data_dir + 'pts2d-norm-pic_a.txt')
    Points_3D = np.loadtxt(data_dir + 'pts3d-norm.txt')

    # (Optional) Run this once you have your code working
    # with the easier, normalized points above.
    if args.hard_points:
        Points_2D = np.loadtxt(data_dir + 'pts2d-pic_b.txt')
        Points_3D = np.loadtxt(data_dir + 'pts3d.txt')

    # Calculate the projection matrix given corresponding 2D and 3D points
    # !!! You will need to implement calculate_projection_matrix. !!!
    M = calculate_projection_matrix(Points_2D, Points_3D)
    print('The projection matrix is:\n {0}\n'.format(M))

    Projected_2D_Pts, Residual = evaluate_points(M, Points_2D, Points_3D)
    print('The total residual is:\n {0}\n'.format(Residual))

    if not args.no_vis:
        visualize_points(Points_2D, Projected_2D_Pts)

    # Calculate the camera center using the M found from previous step
    # !!! You will need to implement compute_camera_center. !!!
    Center = compute_camera_center(M)
    print('The estimated location of the camera is:\n {0}\n'.format(Center))

    if not args.no_vis:
        plot3dview(Points_3D, Center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hard_points",
        help="Use the harder, unnormalized points, to be used after you have it \
                working with normalized points (for parts 1 and 2)",
        action="store_true")
    parser.add_argument("--no-vis",
                        help="Disable visualization for part 1",
                        action="store_true")
    arguments = parser.parse_args()
    main(arguments)
