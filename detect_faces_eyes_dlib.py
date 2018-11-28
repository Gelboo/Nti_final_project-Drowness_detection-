from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
def get_components_dlib(image):
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	faces = []
	eyes = []
	clone = image.copy()
	for (i, rect) in enumerate(rects):

		top_corner = (rect.left(),rect.top())
		bottom_right = (rect.right(),rect.bottom())
		roi_face = image[top_corner[1]:bottom_right[1], top_corner[0]:bottom_right[0]].copy()
		faces.append(roi_face)
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the face parts individually
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# clone the original image so we can draw on it, then
			# display the name of the face part on the image

			if (name not in ['right_eye','left_eye']) :
				continue
			cv2.rectangle(clone,top_corner,bottom_right,(255,0,0),2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 2, (0, 0, 255), -1)

	            # extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			eyes.append(roi)
	return clone,faces,eyes
