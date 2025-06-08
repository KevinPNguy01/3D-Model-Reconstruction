
from src.camutils import Camera, makerotation
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pickle
import glob
import cv2
import os

def calibrate(calibimgfiles, checkerboard_folder, resultfile):
    """
    Create a camera calibration using a set of images of a checkerboard pattern if a calibration file at resultfile doesn't exist.
    
    Parameters
    ----------
    calibimgfiles : str
        Path to the calibration images (glob pattern).

    checkerboard_folder : str
        Folder containing the checkerboard images. This is used to determine the extrinsic parameters of the cameras.
        
    resultfile : str
        File name to save the calibration results.
    """

    # If the result file already exists, load it
    if glob.glob(resultfile):
        print(f"Calibration file {resultfile} already exists. Loading from file...")
        with open(resultfile, 'rb') as file:
            camL, camR = pickle.load(file)
        return camL, camR
    
    # Otherwise, calibrate the cameras
    print(f"Calibration file {resultfile} does not exist. Creating calibration...")
    f, c = calibrate_intrinsic(calibimgfiles)
    camL = calibrate_extrinsic(f'{checkerboard_folder}/color_C0_00_u.png', f, c)
    camR = calibrate_extrinsic(f'{checkerboard_folder}/color_C1_00_u.png', f, c)

    # Save the calibration results
    folder = os.path.dirname(resultfile)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(resultfile, 'wb') as file:
        pickle.dump((camL, camR), file)
    print(f"Calibration complete. Result saved to {resultfile}.")

    return camL, camR

def calibrate_intrinsic(calibimgfiles):
    """
    Calibrate a camera using a set of images of a checkerboard pattern. The checkerboard should have 7 rows and 9 columns of squares.
    
    Parameters
    ----------
    calibimgfiles : str
        Path to the calibration images (glob pattern).

    Returns
    -------
    f : float
        Focal length of the camera in pixels.

    c : numpy.array (shape=(2,1))
        Principal point of the camera in pixels.

    """

    # checkerboard coordinates in 3D
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = 2.8*np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibimgfiles)

    if len(images)==0:
        print('No images found')
        exit()

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Display image with the corners overlayed
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(50)

    cv2.destroyAllWindows()

    # now perform the calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Compute focal length and principal point
    f = np.mean([K[0][0], K[1][1]])
    c =  np.array([[K[0][2]], [K[1][2]]])

    return f, c


def calibrate_extrinsic(img_path, f, c):
    """
    Calibrate the extrinsic parameters of a camera using a single image of a checkerboard pattern.
    The checkerboard should have 6 rows and 6 columns of squares.

    Parameters
    ----------
    img_path : str
        Path to the calibration image of the checkerboard pattern.

    f : float
        Focal length of the camera in pixels.

    c : numpy.array (shape=(2,1))
        Principal point of the camera in pixels.

    Returns
    -------
    cam : Camera
        Camera object with the calibrated extrinsic parameters.

    """

    def residuals(pts3, pts2, cam, params):
        """
        Compute the difference between the projection of 3D points by the camera
        with the given parameters and the observed 2D locations. Used for least
        squares optimization to refine the camera's extrinsic parameters.

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        pts2 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (2,N)

        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing stored in a vector of shape (6,)

        Returns
        -------
        residual : 1D numpy.array (dtype=float)
            Vector of residual 2D projection errors of size 2*N
            
        """

        cam.update_extrinsics(params)
        pts2proj = cam.project(pts3)
        residual = pts2proj - pts2
        residual = residual.flatten()

        return residual
    
    # Calculate the known 3D points of the checkerboard pattern
    # The checkerboard has 6 rows and 6 columns of squares, each square is 2.8 cm
    pts3 = np.zeros((3,6*6))
    zz,xx = np.meshgrid(np.arange(6),np.arange(6))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[2,:] = 2.8*zz.reshape(1,-1)
    
    # Read the image and find the corners of the checkerboard pattern
    img = plt.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img * 255).astype(np.uint8)
    _, corners = cv2.findChessboardCorners(img, (6,6), None)
    pts2 = corners.squeeze().T
    
    # Optimize the camera's extrinsic parameters
    cam = Camera(f, c, makerotation(0, 0, 0), np.array([0, 0, 0]))
    func = lambda params: residuals(pts3,pts2,cam,params)
    new_params = scipy.optimize.leastsq(func, [0,0,0,0,0,-25])
    cam.update_extrinsics(new_params[0])

    return cam