from calibrate import create_calibration
from decode import decode
from reconstruct import reconstruct
from visualize import visualize_points, visualize_mesh
from mesh import mesh, writeply
from align import align
import numpy as np
import matplotlib.pyplot as plt

def main():
    camL, camR = create_calibration("./scans/calib/*.png", "./scans/david/grab_0", "./pickles/calibration/cameras.pickle")

    pts = []
    ptsL = []
    colors = []
    triangles = []
    for i in range(5):
        C0, M0, C1, M1 = decode(f"./scans/david/grab_{i}", 0.005, 0.15, f"./pickles/graycode/david_{i}.pickle")
        pts2L, pts2R, pts3, rgb = reconstruct(f"./scans/david/grab_{i}", C0, C1, M0, M1, camL, camR, f"./pickles/reconstruction/david_{i}.pickle")
        pts3, pts2L, pts2R, rgb, faces = mesh(pts3, pts2L, pts2R, rgb, [-100, 100, 0.25, 100, -100, 100], 5, f"./pickles/mesh/david_{i}.pickle")

        pts.append(pts3)
        ptsL.append(pts2L)
        colors.append(rgb)
        triangles.append(faces)

    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1))]
    for i in range(4):
        j = i + 1
        R, t = align(
            pts[i], pts[j], ptsL[i], ptsL[j],
            f"./scans/david/grab_{i}/color_C0_01_u.png", f"./scans/david/grab_{j}/color_C0_01_u.png", 
            f"./pickles/alignment/david_{i}_{j}.pickle",
        )
        rotations.append(R)
        translations.append(t)

if __name__ == "__main__":
    main()