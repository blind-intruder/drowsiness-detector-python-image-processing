__developers__=["M. Farhan","M. Faizan"]

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
yawns_count = 0
yawn_status = False 
eye_ratio_threshold = 0.25
eye_ratio_frames = 20
counter = 0
isalarmed = False

class detect_drowsiness():

	def get_face(self,img):
		face = detector(img, 1)
		if len(face) > 1:
			return "error"
		if len(face) == 0:
			return "error"
		return np.matrix([[p.x, p.y] for p in predictor(img, face[0]).parts()])

	def get_eye_aspect_ratio(self,eye):
		v1=dist.euclidean(eye[1], eye[5])
		v2=dist.euclidean(eye[2], eye[4])
		v3=dist.euclidean(eye[0], eye[3])

		ratio=(v1 + v2) / (2.0 * v3)
		return ratio

	def mark_face(self,img,marks):
		img = img.copy()
		for i, points in enumerate(marks):
			p = (points[0, 0], points[0, 1])
			cv2.circle(img, p, 3, color=(1, 2, 255))
		return img

	def get_upper_lip(self,marks):
		upper_lip_points = []
		for i in range(50,53):
			upper_lip_points.append(marks[i])
		for i in range(61,64):
			upper_lip_points.append(marks[i])
		upper_lip_total_points = np.squeeze(np.asarray(upper_lip_points))
		upper_lip_mean = np.mean(upper_lip_points, axis=0)
		return int(upper_lip_mean[:,1])

	def get_bottom_lip(self,marks):
		bottom_lip_points = []
		for i in range(65,68):
			bottom_lip_points.append(marks[i])
		for i in range(56,59):
			bottom_lip_points.append(marks[i])
		bottom_lip_total_points = np.squeeze(np.asarray(bottom_lip_points))
		bottom_lip_mean = np.mean(bottom_lip_points, axis=0)
		return int(bottom_lip_mean[:,1])

	def check_mouth_open(self,img):
		marks = self.get_face(img)
		if str(marks) == "error":
			return img, 0

		image_with_marks = self.mark_face(img,marks)
		upper_lip = self.get_upper_lip(marks)
		bottom_lip = self.get_bottom_lip(marks)
		total_lip_distance = abs(upper_lip - bottom_lip)
		return image_with_marks, total_lip_distance

	def start_detecting(self):
		global counter
		Yawns=0
		counter=0
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		print("trying to start camera")
		try:
			video_input = VideoStream(0).start()
			time.sleep(1.0)
			print("Camera is running")
			print("Detecting Drowsiness")
			Yawns=0
			while 1:
				frames = video_input.read()
				frames = imutils.resize(frames, width=650)
				gray_video = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

				faces = detector(gray_video, 0)
				image_with_marks, total_lip_distance = self.check_mouth_open(frames)
				for face in faces:
					s = predictor(gray_video, face)
					s = face_utils.shape_to_np(s)

					left_eye = s[lStart:lEnd]
					right_eye = s[rStart:rEnd]
					left_eye_ratio = self.get_eye_aspect_ratio(left_eye)
					right_eye_ratio = self.get_eye_aspect_ratio(right_eye)

					eyes_average_ratio=(left_eye_ratio + right_eye_ratio) / 2.0

					left_eye_hull = cv2.convexHull(left_eye)
					right_eye_hull = cv2.convexHull(right_eye)

					cv2.putText(frames, str(eyes_average_ratio), (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					if eyes_average_ratio<eye_ratio_threshold:
						counter+=1

						if counter>=eye_ratio_frames:
							cv2.putText(frames, "DROWSINESS Detected! (Eyes are closed)", (10, 200),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					else:
						counter=0

					image_with_marks, total_lip_distance = self.check_mouth_open(frames)
					if total_lip_distance > 18:
						yawn_status=True
						cv2.putText(image_with_marks, "Yawning Detected!", (10, 250),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					else:
						yawn_status=False	


				cv2.imshow("Drowsiness Detector", image_with_marks)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
			cv2.destroyAllWindows()
			video_input.stop()
		except Exception as e:
			print("Unable to start camera!\n"+str(e))


d=detect_drowsiness()
d.start_detecting()
