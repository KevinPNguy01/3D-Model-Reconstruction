
from src.camutils import Camera, makerotation
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import pickle
import glob
import cv2
import os

def create_calibration(calibimgfiles, checkerboard_folder, resultfile):
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
    if not glob.glob(resultfile):
        print(f"Calibration file {resultfile} does not exist. Creating calibration...")
        folder = os.path.dirname(resultfile)
        if not os.path.exists(folder):
            os.makedirs(folder)
        camL, camR = calibrate_cameras(calibimgfiles, checkerboard_folder, resultfile)
        print(f"Calibration complete. Result saved to {resultfile}.")
    else:
        print(f"Calibration file {resultfile} already exists. Loading from file...")
        with open(resultfile, 'rb') as file:
            camL, camR = pickle.load(file)
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

    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated radial distortion coefficients")
    print(dist)
    print()

    print("Individual intrinsic parameters")
    print("fx = ",K[0][0])
    print("fy = ",K[1][1])
    print("cx = ",K[0][2])
    print("cy = ",K[1][2])
    print()

    # Compute focal length and principal point
    f = np.mean([K[0][0], K[1][1]])
    c =  np.array([[K[0][2]], [K[1][2]]])

    return f, c

def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

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

def calibrate_pose(pts3,pts2,cam,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera
        
    params_init : 1D numpy.array (dtype=float)
        Initial estimate of camera extrinsic parameters ()
        params[0:3] are the rotation angles, params[3:6] are the translation

    Returns
    -------
    cam : Camera
        Refined estimate of camera with updated R,t parameters

    """
    func = lambda params: residuals(pts3,pts2,cam,params)
    new_params = scipy.optimize.leastsq(func,params_init)
    cam.update_extrinsics(new_params[0])
    return cam

def calibrate_extrinsic(folder, f, c):
    """
    Calibrate the extrinsic parameters of two cameras using a checkerboard pattern.
    This function assumes the checkerboard has 7 rows and 7 columns of squares, and each square is 2.8 cm on a side.

    Parameters
    ----------
    folder : str
        Folder containing the left and right images of the checkerboard pattern.
        The images should be named 'color_C0_00_u.png' for the left camera and
        'color_C1_00_u.png' for the right camera.
        
    f : float
        Focal length of the camera in pixels.

    c : numpy.array (shape=(2,1))
        Principal point of the camera in pixels, typically the center of the image.

    Returns
    -------
    camL : Camera
        Camera object for the left camera with updated extrinsic parameters.

    camR : Camera
        Camera object for the right camera with updated extrinsic parameters.

    """
    # load in the left and right images and find the coordinates of
    # the chessboard corners using OpenCV
    imgL = plt.imread(f'{folder}/color_C0_00_u.png')
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgL = (imgL * 255).astype(np.uint8)
    ret, cornersL = cv2.findChessboardCorners(imgL, (6,6), None)
    pts2L = cornersL.squeeze().T

    imgR = plt.imread(f'{folder}/color_C1_00_u.png')
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    imgR = (imgR * 255).astype(np.uint8)
    ret, cornersR = cv2.findChessboardCorners(imgR, (6,6), None)
    pts2R = cornersR.squeeze().T

    # generate the known 3D point coordinates of points on the checkerboard in cm
    pts3 = np.zeros((3,6*6))
    zz,xx = np.meshgrid(np.arange(6),np.arange(6))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[2,:] = 2.8*zz.reshape(1,-1)

    # Now use your calibratePose function to get the extrinsic parameters
    # for the two images. You may need to experiment with the initialization
    # in order to get a good result

    camL = Camera(f, c, makerotation(0, 0, 0), np.array([0, 0, 0]))
    camR = Camera(f, c, makerotation(0, 0, 0), np.array([0, 0, 0]))

    camL_params_init = np.array([0,0,0,0,0,-5])
    camR_params_init = np.array([0,0,0,0,0,-5])

    camL = calibrate_pose(pts3,pts2L,camL,camL_params_init)
    camR = calibrate_pose(pts3,pts2R,camR,camR_params_init)

    print("Estimated camera extrinsic parameters for left camera:")
    print("Rotation matrix R:")
    print(camL.R)
    print("Translation vector t:")
    print(camL.t)
    print()

    print("Estimated camera extrinsic parameters for right camera:")
    print("Rotation matrix R:")
    print(camR.R)
    print("Translation vector t:")
    print(camR.t)

    return camL, camR

def calibrate_cameras(calibimgfiles, checkerboard_folder, resultfile):
    """
    Calibrate the cameras using a set of images of a checkerboard pattern.
    
    Parameters
    ----------
    calibimgfiles : str
        Path to the calibration images (glob pattern).
        
    checkerboard_folder : str
        Folder containing the checkerboard images. This is used to determine the extrinsic parameters of the cameras.
        
    resultfile : str
        File name to save the calibration results.

    Returns
    -------
    camL : Camera
        Camera object for the left camera with intrinsic and extrinsic parameters.

    camR : Camera
        Camera object for the right camera with intrinsic and extrinsic parameters.

    """
    f, c = calibrate_intrinsic(calibimgfiles)

    camL, camR = calibrate_extrinsic(checkerboard_folder, f, c)

    # Save the calibration results
    with open(resultfile, 'wb') as file:
        pickle.dump((camL, camR), file)

    return camL, camR