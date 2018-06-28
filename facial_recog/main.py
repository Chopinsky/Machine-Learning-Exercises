from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--dataset", required=True,
                    help="path to input directory of faces + images")
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model to use: either `hog` or `cnn`")

    return vars(ap.parse_args())


def process_images(image_paths):
    known_encodings = []
    known_names = []

    # process the images and generate face-encodings with labels
    for (i, imagePath) in enumerate(image_paths):
        print("info: processing image {}/{}".format(i+1, len(image_paths)))

        # prepare the image and the label
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # identify the face area, and generate the face encodings
        boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)

        # save the encoding and the label
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    return {"encodings": known_encodings, "names": known_names}


def save_to_file(path, image_data):
    f = open(path, "wb")
    f.write(pickle.dumps(image_data))
    f.close()


if __name__ == '__main__':
    args = parse_args()
    imagePaths = list(paths.list_images(args["dataset"]))

    print("info: quantifying faces...")
    data = process_images(imagePaths)

    print("info: serializing encodings...")
    save_to_file(args["encodings"], data)
