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


def process_image(args):
    print("info: processing image...")
    data = pickle.loads(open(args["encodings"], "rb").read())

    image = cv2.imread(args["image"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            indices = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in indices:
                name = data["name"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    return names, boxes


if __name__ == '__main__':
    args = parse_args()
    names, boxes = process_image(args)
