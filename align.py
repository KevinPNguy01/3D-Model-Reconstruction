import cv2
import numpy as np
import glob
import os
import pickle

def select_points(img1_path, img2_path):
    """
    Have the user click two corresponding points in two images.

    Parameters
    ----------
    img1_path : str
        Path to the first image.

    img2_path : str
        Path to the second image.

    Returns
    -------
    points1 : list of tuples
        List of (x, y) coordinates of points selected in the first image.

    points2 : list of tuples
        List of (x, y) coordinates of points selected in the second image.
    
    """

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    points1 = []
    points2 = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            window_name, img, points = param
            points.append((x, y))
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(img, f"{len(points)}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow(window_name, img)

    cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image 1', 600, 400)
    cv2.resizeWindow('Image 2', 600, 400)

    cv2.imshow('Image 1', img1)
    cv2.setMouseCallback('Image 1', click_event, ("Image 1", img1, points1))
    cv2.imshow('Image 2', img2)
    cv2.setMouseCallback('Image 2', click_event, ("Image 2", img2, points2))

    while True:
        if cv2.waitKey(50) != -1:
            break

        if cv2.getWindowProperty('Image 1', cv2.WND_PROP_VISIBLE) < 1 and \
           cv2.getWindowProperty('Image 2', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    if len(points1) != len(points2) or len(points1) < 2:
        raise ValueError("Please select at least two corresponding points in both images.")

    return points1, points2

def align(pts3_1, pts3_2, ptsL1, ptsL2, img1_path, img2_path, resultfile):
    """
    Align two sets of 3D points based on selected corresponding points in two images.

    Parameters
    ----------
    pts3_1 : 2D numpy array
        3D points from the first image, shape (3, N).

    pts3_2 : 2D numpy array
        3D points from the second image, shape (3, N).

    ptsL1 : 2D numpy array
        2D points in the left camera for the first image, shape (2, N).

    ptsL2 : 2D numpy array
        2D points in the left camera for the second image, shape (2, N).

    img1_path : str
        Path to the first image.

    img2_path : str
        Path to the second image.

    resultfile : str
        Path to the file where the results will be saved.

    Returns
    -------
    R : 2D numpy array
        Rotation matrix aligning the points from the second image to the first image.

    t : 1D numpy array
        Translation vector aligning the points from the second image to the first image.

    """

    if glob.glob(resultfile):
        print(f"Loading alignment results from {resultfile}...")
        with open(resultfile, 'rb') as f:
            R, t = pickle.load(f)
        return R, t

    pts1, pts2 = select_points(img1_path, img2_path)

    nearest_ptsL1_indices = []
    nearest_ptsL2_indices = []
    for index in range(len(pts1)):
        pt1, pt2 = pts1[index], pts2[index]
        nearest_ptsL1_indices.append(0)
        nearest_ptsL2_indices.append(0)

        for k in range(len(ptsL1[0])):
            if np.linalg.norm(ptsL1[:, k] - pt1) < np.linalg.norm(ptsL1[:, nearest_ptsL1_indices[-1]] - pt1):
                nearest_ptsL1_indices[-1] = k
        for k in range(len(ptsL2[0])):
            if np.linalg.norm(ptsL2[:, k] - pt2) < np.linalg.norm(ptsL2[:, nearest_ptsL2_indices[-1]] - pt2):
                nearest_ptsL2_indices[-1] = k

    nearest_pts3_1 = pts3_1[:, nearest_ptsL1_indices]
    nearest_pts3_2 = pts3_2[:, nearest_ptsL2_indices]

    m1 = np.mean(nearest_pts3_1, axis=1)
    m2 = np.mean(nearest_pts3_2, axis=1)
    X1c = nearest_pts3_1 - m1[:, np.newaxis]
    X2c = nearest_pts3_2 - m2[:, np.newaxis]
    H = X1c @ X2c.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    R = R.T
    t = m1 - R @ m2

    folder = os.path.dirname(resultfile)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(resultfile, 'wb') as f:
        pickle.dump((R, t), f)

    return R, t