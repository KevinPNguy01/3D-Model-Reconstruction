from calibrate import create_calibration
from decode import decode_combined_code
from reconstruct import reconstruct
from visualize import visualize_points
import pickle

def main():
    create_calibration("./calib/*.png", "./david/grab_0", "calibration.pickle")

    C0, M0 = decode_combined_code("./david/grab_3", "C0", 0.005, 0.15)
    C1, M1 = decode_combined_code("./david/grab_3", "C1", 0.005, 0.15)

    with open("calibration.pickle", "rb") as f:
        camL, camR = pickle.load(f)

    pts2L, pts2R, pts3, rgb = reconstruct("./david/grab_3/color_C0_01_u.png", "./david/grab_3/color_C1_01_u.png", C0, C1, M0, M1, camL, camR)
    visualize_points(pts3, rgb)

if __name__ == "__main__":
    main()