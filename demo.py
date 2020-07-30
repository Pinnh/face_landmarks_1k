import torch
from face_detector import *
from face_landmark import *
import sys

def camera_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if image is None:
            continue
        detections, _ = face_detector_handle.run(image)

        if len(detections) == 0:
            continue
        for detection in detections:
            landmarks, states = face_landmark_handle.run(image, detection)
            if landmarks is None:
                continue
            face_landmark_handle.show_result(image, landmarks)


def image_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    image = cv2.imread(sys.argv[1])
    print("det:",sys.argv[1])
    print("shape:",image.shape)
    detections, _ = face_detector_handle.run(image)
    print(detections)
    #face_detector_handle.show_result(image, detections)
    print("det:",len(detections))
    if len(detections) == 0:
        return

    for detection in detections:
        print("get landmarks")
        landmarks, states = face_landmark_handle.run(image, detection)
        face_landmark_handle.show_result(image, landmarks)


if __name__ == '__main__':
    image_run()
