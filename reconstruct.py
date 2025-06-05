import numpy as np
import matplotlib.pyplot as plt

def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    #
    # Your code goes here.  I recommend adding assert statements to check the
    # sizes of the inputs and outputs to make sure they are correct 
    #

    pts3 = np.zeros((3, pts2L.shape[1]))

    for i in range(pts2L.shape[1]):
        qL = np.vstack(((pts2L[:, i].reshape(-1, 1) - camL.c) / camL.f, np.array([[1]])))
        qR = np.vstack(((pts2R[:, i].reshape(-1, 1) - camR.c) / camR.f, np.array([[1]])))
        A = np.hstack((camL.R @ qL, -camR.R @ qR))
        b = camR.t - camL.t
        u = np.linalg.lstsq(A, b, rcond=None)[0]
        
        pL = u[0] * qL
        pR = u[1] * qR
        p1 = camL.R @ pL + camL.t
        p2 = camR.R @ pR + camR.t
        pts3[:, i] = (p1 + p2).flatten() / 2

    return pts3

def reconstruct(img0_path, img1_path, C0, C1, M0, M1, camL, camR):
    """
    Reconstruct the 3D points from the codes C0 and C1 for the left and right cameras.


    Parameters
    ----------
    img0_path : str
        Path to the left camera image

    img1_path : str
        Path to the right camera image

    C0 : 2D numpy.array (dtype=float)
        The codes for the left camera, with shape (H,W)

    C1 : 2D numpy.array (dtype=float)
        The codes for the right camera, with shape (H,W)

    M0 : 2D numpy.array (dtype=bool)
        The mask for the left camera, with shape (H,W)

    M1 : 2D numpy.array (dtype=bool)
        The mask for the right camera, with shape (H,W)

    camL : Camera
        The left camera object containing calibration data

    camR : Camera
        The right camera object containing calibration data


    Returns
    -------
    pts2L : 2D numpy.array (dtype=float)
        The pixel coordinates in the left camera for the matched points, shape (2,N)

    pts2R : 2D numpy.array (dtype=float)
        The pixel coordinates in the right camera for the matched points, shape (2,N)

    pts3 : 2D numpy.array (dtype=float)
        The 3D coordinates of the matched points in the global coordinate system, shape (3,N)
    """
    
    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    _, matchL, matchR = np.intersect1d(C0[M0], C1[M1], return_indices=True)
    
    # Let CL and CR be the flattened arrays of codes for the left and right view
    # Suppose you have computed arrays of indices matchL and matchR so that 
    # CL[matchL[i]] == CR[matchR[i]] for all i.  The code below gives one approach
    # to generating the corresponding pixel coordinates for the matched pixels.

    h,w = C0.shape    
    xx,yy = np.meshgrid(range(w),range(h))
    xxL = xx[M0]
    yyL = yy[M0]
    xxR = xx[M1]
    yyR = yy[M1]
    pts2L = np.vstack((xxL[matchL], yyL[matchL]))
    pts2R = np.vstack((xxR[matchR], yyR[matchR]))
 
    # Now triangulate the points

    pts3 = triangulate(pts2L, camL, pts2R, camR)
    pts3[1, :] *= -1

    img0 = plt.imread(img0_path)
    img1 = plt.imread(img1_path)

    rgb_values = img0[pts2L[1], pts2L[0]].T
    
    return pts2L, pts2R, pts3, rgb_values