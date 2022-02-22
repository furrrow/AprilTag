#!/usr/bin/env python
from argparse import ArgumentParser
import os
import cv2
import apriltag
import pickle

"""
result: list of detections
detection.tag_id
detection.tag_family
detection.decision_margin
detection.goodness
detection.hamming
detection.homography (3x3 array)
detection.corners (list of 4 coordinates (4, 2))
detection.center (2,)
detection.tostring
"""

def opencv_script():
    # IMG_LOC = "/root/Downloads/robots.png"
    # IMG_LOC = "/root/AprilTag/media/input/single_tag.jpg"
    IMG_LOC = "/root/AprilTag/media/test_img/42in.jpg"
    IMAGE_SHOW = True
    parser = ArgumentParser(description='Detect AprilTags from static images.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()
    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())
    camera_file = open("/root/AprilTag/scripts/camera_parameters.p", "rb")
    camera_params = pickle.load(camera_file)
    fx = camera_params['mtx'][0, 0]
    fy = camera_params['mtx'][1, 1]
    cx = camera_params['mtx'][0, 2]
    cy = camera_params['mtx'][1, 2]

    img = cv2.imread(IMG_LOC)

    result, overlay = apriltag.detect_tags(img,
                                            detector,
                                            camera_params=(fx, fy, cx, cy),
                                            tag_size=0.021,
                                            vizualization=3,
                                            verbose=3,
                                            annotation=True
                                            )

    print("result", result)
        # result has 4 entries per detection
        # each detection has array, lat long, etc.
    # print("overlay", overlay)

    if IMAGE_SHOW:
        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.imshow("window", overlay)
        while cv2.waitKey(5) < 0:   # Press any key to load subsequent image
                pass


if __name__ == '__main__':
    opencv_script()
