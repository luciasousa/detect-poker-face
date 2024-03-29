import cv2
from keras import backend 
from tensorflow.keras.models import load_model
import numpy as np
import dlib

user_input = input("Please enter a string: ")

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))

# Preprocessing block
def hist_eq(img):
	#call the .apply method on the CLAHE object to apply histogram equalization
    return clahe.apply(img)

def smooth(img):
	#Gaussian blurring is highly effective in removing Gaussian noise from an image.
	return cv2.GaussianBlur(img,(3,3),0)

def resize(img):
    resized_img = cv2.resize(img, (299, 299))
    resized_img = np.repeat(resized_img[..., np.newaxis], 3, axis=-1)
    return resized_img

# Rotation correction
def rotate(gray_image, shape):
	dY = shape[36][1] - shape[45][1]
	dX = shape[36][0] - shape[45][0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	rows,cols = gray_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(gray_image,M,(cols,rows))

	#transform points
	ones = np.ones(shape=(len(shape), 1))
	points_ones = np.hstack([shape, ones])
	new_shape  = M.dot(points_ones.T).T
	new_shape = new_shape.astype(int)

	return dst, new_shape

# Crop the face by using the facial landmarks of dlib
def cropping(rotated_img, shape):
	aux = shape[4] - shape[12]
	distance = np.linalg.norm(aux)
	h = int(distance * 0.1)
	tl = (int((shape[36][0]+shape[18][0])/2), shape[18][1]-h)
	br = (int((shape[45][0]+shape[25][0])/2), int((shape[57][1]+(shape[10][1]+shape[11][1])/2)/2))
	roi = rotated_img[tl[1]:br[1],tl[0]:br[0]]
	return roi

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	
	return (x, y, w, h)

def shape_to_np(shape):
	landmarks = np.zeros((68,2), dtype = int)
	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)

	return landmarks

def detect_face(image, gray_image):
	rects = detector(gray_image, 1)
	shape = []
	bb = []

	for (z, rect) in enumerate(rects):
		if rect is not None and rect.top() >= 0 and rect.right() <= gray_image.shape[1] and rect.bottom() <= gray_image.shape[0] and rect.left() >= 0:
			predicted = predictor(gray_image, rect)
			shape.append(shape_to_np(predicted))
			(x, y, w, h) = rect_to_bb(rect)
			bb.append((x, y, w, h))
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	return shape, bb

def get_emotion(label):
	if label == 0:
		return "Neutral"
	elif label == 1:
		return "Not Neutral"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

video_capture = cv2.VideoCapture(0)

model = load_model('../../inceptionv3.h5')
filtered_face = np.zeros((1, 32, 32, 3))

highest_neutral_confidence = 0.0
frame_with_highest_confidence = None

while True:
	ret, frame = video_capture.read()
	if ret:
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		shape, bb = detect_face(frame, gray_image)
		for i in range(0,len(shape)):
			face = []
			rotated_img, landmarks = rotate(gray_image, shape[i])
			cropped_face = cropping(rotated_img, landmarks)
			eq_face = hist_eq(cropped_face)
			filtered_face = smooth(eq_face)
			resized_face = resize(filtered_face)
			face.append(resized_face)
			face = np.array(face)
			face = face.reshape(face.shape[0], face.shape[1], face.shape[2], 3)
			face = face.astype('float32') / 255
			prediction = model.predict(face)
			emotion = get_emotion(np.argmax(prediction))
			confidence = prediction[0][0]
			if confidence > highest_neutral_confidence and emotion == 'Neutral':
				highest_neutral_confidence = confidence
				frame_with_highest_confidence = frame.copy()
			x = bb[i][0]-10
			y = bb[i][1]+bb[i][2]+20
			print(f"Emotion: {emotion}, Confidence: {confidence}")
			cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

		cv2.imshow('Frame', frame)
		cv2.imshow('filtered_face', filtered_face)
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

# Save the frame with the highest confidence as "neutral_frame.jpg"
if frame_with_highest_confidence is not None:
    cv2.imwrite("../../my_dataset/neutral_" + str(int(highest_neutral_confidence*100))+ "_" + str(user_input) + ".jpg", frame_with_highest_confidence)

video_capture.release()
cv2.destroyAllWindows()
backend.clear_session()