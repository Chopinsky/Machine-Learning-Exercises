from imutils.video import VideoStream
from facial_recog.face_recog import process_image, mark_faces
import imutils
import argparse
import pickle
import time
import cv2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", required=True,
                    help="path to serialized db of facial encodings")
    ap.add_argument("-o", "--output", type=str,
                    help="path to the output video")
    ap.add_argument("-y", "--display", type=int, default=1,
                    help="if to display the output frame to screen")
    ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                    help="face detection model o use: 'hog' or 'cnn'")

    return vars(ap.parse_args())


def process_video(args, write_to_disk=True):
    print("info: loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())

    print("info: starting video stream...")
    vs = VideoStream(src=0).start()
    writer = None

    time.sleep(2.0)

    while True:
        frame = vs.read()
        names, boxes = process_frame(data, frame)
        mark_faces(frame, names, boxes)

        if write_to_disk:
            if writer is None and args["output"] is not None:
                writer = init_videowriter(args["output"], frame.shape[1], frame.shape[0])

            if writer is not None:
                writer.write(frame)

        if args["display"] > 0:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if key "q" is pressed, quit the loop, i.e. the video stream
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()

    if writer is not None:
        writer.release()


def process_frame(data, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        return process_image(args, data, rgb)


def init_videowriter(output, height, width):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output, fourcc, 20, (height, width), True)


if __name__ == '__main__':
    args = parse_args()
    process_video(args)
