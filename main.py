from calibrate import create_calibration
from decode import decode_combined_code
from reconstruct import reconstruct
from visualize import visualize_points, visualize_mesh
from mesh import generate_mesh
import pickle

def main():
    create_calibration("./calib/*.png", "./david/grab_0", "calibration.pickle")

    C0, M0 = decode_combined_code("./david/grab_0", "C0", 0.005, 0.15)
    C1, M1 = decode_combined_code("./david/grab_0", "C1", 0.005, 0.15)

    with open("calibration.pickle", "rb") as f:
        camL, camR = pickle.load(f)

    pts2L, pts2R, pts3, rgb = reconstruct("./david/grab_0/color_C0_01_u.png", "./david/grab_0/color_C1_01_u.png", C0, C1, M0, M1, camL, camR)
    visualize_points(pts3, rgb)

    pts3, rgb, faces = generate_mesh(pts3, pts2L, rgb, [-100, 100, 0.1, 100, -100, 100], 2)
    visualize_mesh(pts3, rgb, faces)

if __name__ == "__main__":
    main()