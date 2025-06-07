from calibrate import create_calibration
from decode import decode
from reconstruct import reconstruct
from visualize import visualize_points, visualize_mesh
from mesh import mesh, writeply
from align import align
import numpy as np
import matplotlib.pyplot as plt

def main():
    camL, camR = create_calibration(".scans/calib/*.png", "./scans/david/grab_0", "./pickles/calibration/cameras.pickle")

    pts = []
    ptsL = []
    colors = []
    triangles = []

    for i in range(5):

        C0, M0, C1, M1 = decode(f"./scans/david/grab_{i}", 0.005, 0.25, f"./pickles/graycode/david_{i}.pickle")

        pts2L, pts2R, pts3, rgb = reconstruct(f"./scans/david/grab_{i}", C0, C1, M0, M1, camL, camR, f"./pickles/reconstruction/david_{i}.pickle")
        #visualize_points(pts3, rgb)
        plt.show()

        pts3, pts2L, pts2R, rgb, faces = mesh(pts3, pts2L, pts2R, rgb, [-100, 100, 0.25, 100, -100, 100], 5, f"./pickles/mesh/david_{i}.pickle")
        #visualize_mesh(pts3, rgb, faces)

        pts.append(pts3)
        ptsL.append(pts2L)
        colors.append(rgb)
        triangles.append(faces)

if __name__ == "__main__":
    main()