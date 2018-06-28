import face_recognition
import argparse
import pickle
import cv2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model o use: 'hog' or 'cnn'")

    return vars(ap.parse_args())


if __name__ == '__main__':
    args = parse_args()
