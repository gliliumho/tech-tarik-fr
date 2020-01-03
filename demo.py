#! python3

import os
from typing import List
import cv2
import numpy as np
import tensorflow as tf

from sklearn.neighbors import BallTree
from sklearn import linear_model
import joblib

import align.detect_face
from facenet import facenet


VIDEO_INPUT = 0
FACE_MODEL = 'models/20180402-114759.pb'
FACE_DETECT_MINSIZE = 10
MTCNN_THRESHOLD = [0.7, 0.7, 0.7]
MTCNN_FACTOR = 0.3

FACE_PREDICT_IMGSIZE = 160
FACE_PREDICT_THRESHOLD = 0.5

FACE_PREDICT_MODEL_PATH = 'models/prediction_model.pkl'

SHOW_FACE_IMAGE_FLAG = True


class Face:
    face_image = None
    face_feature = []
    name = ""

    def __init__(self, name, face_image, face_feature):
        self.name = name
        self.face_feature = face_feature
        self.face_image = face_image


class FaceDetection:
    x1, x2, y1, y2 = 0, 0, 0, 0
    face_image = None
    face_feature = []
    name = ""

    def __init__(self, cord: np.array, rgb_frame):
        self.x1 = int(cord[0])
        self.y1 = int(cord[1])
        self.x2 = int(cord[2])
        self.y2 = int(cord[3])
        self.face_image = rgb_frame[self.y1:self.y2, self.x1:self.x2]

    def get_face_image(self):
        return self.face_image

    def set_face_feature(self, face_feature: List[float]):
        self.face_feature = face_feature

    def get_face_feature(self):
        return self.face_feature

    def set_name(self, name: str):
        self.name = name

    def get_name(self):
        return self.name


class FR:
    pnet = None
    rnet = None
    onet = None
    fnet = None

    predict_model = None
    registered_name_list = []

    def __init__(self):
        self.load_networks()
        self.load_prediction_model()

    def load_networks(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions()
            sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(
                    sess, None)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions()
            sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                facenet.load_model(FACE_MODEL)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.fnet = lambda images: sess.run(embeddings, feed_dict={
                                                    images_placeholder: images, phase_train_placeholder: False})

    def load_prediction_model(self):
        if os.path.exists(FACE_PREDICT_MODEL_PATH):
            self.predict_model, self.registered_name_list = joblib.load(
                FACE_PREDICT_MODEL_PATH)
        else:
            print("Prediction model does not exist, no registered faces.")
            # self.predict_model = BallTree(np.empty([0, 64], dtype=np.float))
            # joblib.dump((self.predict_model, []), FACE_PREDICT_MODEL_PATH)

    def detect_faces(self, full_frame: np.array):
        face_locations = align.detect_face.detect_face(
            full_frame, FACE_DETECT_MINSIZE, self.pnet, self.rnet, self.onet, MTCNN_THRESHOLD, MTCNN_FACTOR)

        face_detections = []
        img_size = np.asarray_chkfinite(full_frame.shape)
        for faceloc in face_locations[0]:
            face_det = FaceDetection(faceloc[0:4], full_frame)
            face_detections.append(face_det)

        if SHOW_FACE_IMAGE_FLAG:
            for i, face_det in enumerate(face_detections):
                cv2.imshow('face_image %d' % i, face_det.get_face_image())

        return face_detections

    def get_face_features(self, face_detections: List[FaceDetection]):
        prewhitened_face_images = []
        for face_det in face_detections:
            face_img = cv2.resize(
                face_det.get_face_image(), (FACE_PREDICT_IMGSIZE, FACE_PREDICT_IMGSIZE))
            prewhitened = facenet.prewhiten(face_img)
            prewhitened_face_images.append(prewhitened)

        prewhitened_face_images = np.stack(prewhitened_face_images)

        face_features = self.fnet(prewhitened_face_images)
        face_features = np.array(
            [f[12::14]/np.linalg.norm(f[12::14]) for f in face_features])

        print("Face detections: %d" % len(face_detections))
        print("Face features: %d" % len(face_features))

        for i, face_det in enumerate(face_detections):
            face_det.set_face_feature(face_features[i])

        return face_features


def main():
    vid_capture = cv2.VideoCapture(VIDEO_INPUT)
    if vid_capture.isOpened() == False:
        print("Error opening %s" % (VIDEO_INPUT))
    fr = FR()
    fr.load_networks()

    while True:
        ret, bgr_frame = vid_capture.read()

        if ret:
            rgb_frame = bgr_frame[:, :, ::-1]
            face_detections = fr.detect_faces(rgb_frame)
            if face_detections:
                face_features = fr.get_face_features(face_detections)

            print(len(face_detections))

            cv2.imshow('FR demo', bgr_frame)
            cv2.waitKey(100)


if __name__ == '__main__':
    main()
