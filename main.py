from src import calibrate, decode, reconstruct, mesh, writeply, align
import numpy as np
import sys

def main(name, n):
    """
    Main function to reconstruct a 3D mesh from a sequence of scans.

    Parameters
    ----------
    name : str
        The name of the scan sequence to process. This is used to locate the scan files and save results. e.g., 'david'.

    n : int
        The number of scans to process. Each scan is expected to be named as 'grab_0', 'grab_1', ..., 'grab_n-1'.

    """

    # Calibrate the cameras
    camL, camR = calibrate("./scans/calib/*.png", f"./scans/{name}/grab_0", f"./pickles/calibration/{name}.pickle")

    # Generate partial meshes from each of the scans
    pts = []
    ptsL = []
    colors = []
    triangles = []
    for i in range(n):
        C0, M0, C1, M1 = decode(f"./scans/{name}/grab_{i}", 0.005, 0.15, f"./pickles/graycode/{name}_{i}.pickle")
        pts2L, pts2R, pts3, rgb = reconstruct(f"./scans/{name}/grab_{i}", C0, C1, M0, M1, camL, camR, f"./pickles/reconstruction/{name}_{i}.pickle")
        pts3, pts2L, pts2R, rgb, faces = mesh(pts3, pts2L, pts2R, rgb, [-100, 100, 0.25, 100, -100, 100], 2.5, f"./pickles/mesh/{name}_{i}.pickle")
        pts.append(pts3)
        ptsL.append(pts2L)
        colors.append(rgb)
        triangles.append(faces)

    # Calculate the alignment between each consecutive pair of scans
    rotations = [np.eye(3)]
    translations = [np.zeros((3,))]
    for i in range(n-1):
        j = i + 1
        R, t = align(
            pts[i], pts[j], ptsL[i], ptsL[j],
            f"./scans/{name}/grab_{i}/color_C0_01_u.png", f"./scans/{name}/grab_{j}/color_C0_01_u.png",
            f"./pickles/alignment/{name}_{i}_{j}.pickle",
        )
        rotations.append(R)
        translations.append(t)
        
    # Compose the transformations and save aligned meshes
    R = np.eye(3)
    t = np.zeros((3,))
    for i in range(n):
        t = R @ translations[i] + t
        R = rotations[i] @ R
        writeply(R @ pts[i] + t[:, np.newaxis], colors[i], triangles[i], f"./meshes/{name}_{i}.ply")

if __name__ == "__main__":
    # By default, run the main function with 'david' and 5 scans
    if len(sys.argv) != 3:
        main("david", 5)
        exit(0)

    # If command line arguments are provided, use them
    name = sys.argv[1]
    n = int(sys.argv[2])
    main(name, n)