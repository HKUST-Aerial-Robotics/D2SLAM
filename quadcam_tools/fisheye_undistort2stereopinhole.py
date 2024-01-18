import cv2 as cv
import numpy as np
from transformations import *
from utils.fisheye_undist import *

if __name__ == "__main__":
    # Test code
    import argparse
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input image file")
    parser.add_argument("-f","--fov", type=float, default=190, help="hoizon fov of fisheye")
    args = parser.parse_args()

    K = np.array([[1102.624687198539, 0, 633.6460526237753],
            [0,  1102.5788270373146,  384.79915608735973],
            [0, 0, 1]])
    D = np.array([-0.2269834045157848, 0.35092662127266205, 0.0009542318893859212, -0.0010474398462718213])
    xi = 2.0153496541652567

    undist = FisheyeUndist(K, D, xi, fov=args.fov)
    img = cv.imread(args.input)
    imgs = undist.undistAll(img)
    show = imgs[0]
    for i in range(1, len(imgs)):
        show = cv.hconcat([show, imgs[i]])
    cv.namedWindow("raw", cv.WINDOW_NORMAL|cv.WINDOW_GUI_EXPANDED)
    cv.imshow("raw", img)
    cv.namedWindow("Undist", cv.WINDOW_NORMAL|cv.WINDOW_GUI_EXPANDED)
    cv.imshow("Undist", show)
    cv.write("undistort_stereo_pinhole.jpg", show)
    cv.waitKey(0)