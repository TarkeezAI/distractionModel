# Program constructs Concentration Index and returns a classification of engagement.

import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model


class analysis:

    def __init__(self, frame_width, frame_height):
        self.emotion_model = load_model('./util/model/emotion_recognition.h5')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./util/model/shape_predictor_68_face_landmarks.dat")
        self.faceCascade = cv2.CascadeClassifier(
            './util/model/haarcascade_frontalface_default.xml')
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get_midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(
            eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(
            eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.get_midpoint(facial_landmarks.part(
            eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.get_midpoint(facial_landmarks.part(
            eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_length = hypot(
            (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot(
            (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = ver_line_length / hor_line_length
        return ratio

    # Gaze detection function
    def get_gaze_ratio(self, frame, eye_points, facial_landmarks, gray):
        left_eye_region = np.array([(facial_landmarks.part(pt).x, facial_landmarks.part(pt).y) for pt in eye_points[:6]], np.int32)
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        left_eye_rect = cv2.boundingRect(left_eye_region)
        left_eye_image = eye[left_eye_rect[1]: left_eye_rect[1] + left_eye_rect[3], left_eye_rect[0]: left_eye_rect[0] + left_eye_rect[2]]
        _, threshold_eye = cv2.threshold(left_eye_image, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)
        up_side_threshold = threshold_eye[0: int(height / 2), 0: int(width / 2)]
        up_side_white = cv2.countNonZero(up_side_threshold)
        down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
        down_side_white = cv2.countNonZero(down_side_threshold)
        lr_gaze_ratio = (left_side_white + 10) / (right_side_white + 10)
        ud_gaze_ratio = (up_side_white + 10) / (down_side_white + 10)
        return lr_gaze_ratio, ud_gaze_ratio


    # Main function for analysis

    def detect_face(self, frame):
        # convert the input image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        detected_faces = self.detector(gray_frame)

        # initialize variables for storing engagement scores of each face
        face_scores = []

        # loop over the detected faces
        for face in detected_faces:
            # extract the coordinates of the face region as (x,y,w,h)
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

            # crop the face region from the input frame
            face_region = gray_frame[y:y+h, x:x+w]

            # detect facial landmarks in the face region
            facial_landmarks = self.predictor(gray_frame, face)

            # compute the eye region coordinates from the facial landmarks
            left_eye_points = [36, 37, 38, 39, 40, 41]
            right_eye_points = [42, 43, 44, 45, 46, 47]
            left_eye_region = np.array([(facial_landmarks.part(pt).x, facial_landmarks.part(pt).y) for pt in left_eye_points], np.int32)
            right_eye_region = np.array([(facial_landmarks.part(pt).x, facial_landmarks.part(pt).y) for pt in right_eye_points], np.int32)

            # compute the eye blinking ratio and gaze ratio
            left_eye_blink_ratio = self.get_blinking_ratio(frame, left_eye_points, facial_landmarks)
            right_eye_blink_ratio = self.get_blinking_ratio(frame, right_eye_points, facial_landmarks)
            lr_gaze_ratio, ud_gaze_ratio = self.get_gaze_ratio(frame, left_eye_points, facial_landmarks, gray_frame)

            # classify the engagement score based on the computed ratios
            if left_eye_blink_ratio > 5 and right_eye_blink_ratio > 5:
                face_score = "Distracted"
            elif lr_gaze_ratio < 0.9 and ud_gaze_ratio < 0.7:
                face_score = "Engaged"
            else:
                face_score = "Neutral"

            # store the engagement score of the face
            face_scores.append(face_score)

            # draw bounding box and text label around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, face_score, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # compute the overall engagement score for the input frame
        if face_scores:
            if "Distracted" in face_scores:
                engagement_score = "Distracted"
            elif "Engaged" in face_scores:
                engagement_score = "Engaged"
            else:
                engagement_score = "Neutral"
        else:
            engagement_score = "Neutral"

        return engagement_score, frame

    def process_ci(self):

        # Look at the last 20 concentration indices and if they are all "Pay attention!",
        # then the current concentration index is "Pay attention!", otherwise if the last 10 concentration indices
        # are all "Pay attention!",
        # then the current concentration index is "Distracted!", otherwise the current concentration index is
        # last different concentration index

        ci_history = self.ci[-20:]

        if all(ci == "Pay attention!" for ci in ci_history):
            return "Pay attention!"

        ci_history = self.ci[-10:]

        if all(ci == "Pay attention!" for ci in ci_history):
            return "Distracted!"

        return self.ci[-1]
    
    # Function for detecting emotion
    def detect_emotion(self, gray):
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        faces = self.detect_faces(gray)
        if not faces:
            return

        cropped_face = self.crop_face(gray, faces[0])
        test_image = self.preprocess_image(cropped_face)

        probab = self.get_emotion_probabilities(test_image)
        label, predicted_emotion = self.get_predicted_emotion(probab, emotions)

        self.update_emotion(label)

    # Helper function for detect_emotion [1/4] 
    def detect_faces(self, gray):
        return self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100)
        )

    # Helper function for detect_emotion [2/4] 
    def crop_face(self, gray, face):
        x, y, width, height = face
        return gray[y:y + height, x:x + width]

    # Helper function for detect_emotion [3/4] 
    def preprocess_image(self, image):
        image = cv2.resize(image, (48, 48))
        image = image.reshape([-1, 48, 48, 1])
        image = np.multiply(image, 1.0 / 255.0)
        return image

    # Helper function for detect_emotion [4/4] 
    def get_emotion_probabilities(self, image):
        if self.frame_count % 5 != 0:
            return self.last_probab
        probab = self.emotion_model.predict(image)[0] * 100
        self.last_probab = probab
        return probab

    def gen_concentration_index(self):
        emotion_weights = {0: 0.25, 1: 0.3, 2: 0.6, 3: 0.3, 4: 0.6, 5: 0.9}

        gaze_weights = self.calculate_gaze_weights()

        concentration_index = self.calculate_concentration_index(emotion_weights, gaze_weights)

        return self.get_concentration_index_message(concentration_index)


    def calculate_gaze_weights(self):
        if self.size > 0.2 and self.size < 0.3:
            return 1.5
        elif 1 <= self.x <= 5:
            return 2
        else:
            return 1.5

    # Helper function for calculate_gaze_weights [1/2] 
    def calculate_concentration_index(self, emotion_weights, gaze_weights):
        max_weights_product = 4.5
        return (emotion_weights[self.emotion] * gaze_weights) / max_weights_product

    # Helper function for calculate_gaze_weights [2/2] 
    def get_concentration_index_message(self, concentration_index):
        if concentration_index > 0.65:
            return "You are highly focused!"
        elif 0.25 < concentration_index <= 0.65:
            return "You are focused."
        else:
            return "Pay attention!"
