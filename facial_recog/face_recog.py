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


def process_image(args, data, rgb):
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


def mark_faces(image, names, boxes, display_to_screen=False):
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        if (top - 15) > 15:
            label_y = top - 15
        else:
            label_y = top + 15

        cv2.putText(image, name, (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if display_to_screen:
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def load_data():
    print("info: processing image...")
    data = pickle.loads(open(args["encodings"], "rb").read())

    image = cv2.imread(args["image"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, rgb, data


if __name__ == '__main__':
    args = parse_args()

    image, rgb, data = load_data()
    names, boxes = process_image(args, data, rgb)

    mark_faces(image, names, boxes)