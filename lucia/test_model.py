import os
from tensorflow.keras.models import load_model
import cv2
import dlib
import numpy as np

# Convert the facial landmarks dlib format to numpy
def shape_to_np(shape):
	# pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates
	# that map to facial structures on the face
	landmarks = np.zeros((68,2), dtype = int)
	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)
	return landmarks

# take the bounding box predicted by dlib library
# and convert it into (x, y, w, h) where x, y are coordinates
# and w, h are width and height
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

#Draw the contours
def draw_contour(shape):
	convexHull = cv2.convexHull(shape)
	cont = cv2.drawContours(image, [convexHull], -1, (255, 0, 0), 2)
	
	return cont

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

# Crop the face
def crop_face(gray_image, shape):
    aux = shape[4] - shape[12]
    distance = np.linalg.norm(aux)
    h = int(distance * 0.1)
    tl = (int((shape[36][0]+shape[18][0])/2), shape[18][1]-h)
    br = (int((shape[45][0]+shape[25][0])/2), int((shape[57][1]+(shape[10][1]+shape[11][1])/2)/2))
    roi = gray_image[tl[1]:br[1],tl[0]:br[0]]
    return roi

def get_emotion(label):
	if label == 0:
		return "Neutral"
	else:
		return "Not Neutral"
    
#FIRST - detect the face with dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#FOR VIDEO
'''
video_capture = cv2.VideoCapture(0)
model=load_model('../../model_lucia.h5')
model.summary()
filtered_face = np.zeros((1, 48, 48, 1))
while True:
    ret, frame = video_capture.read()
    if ret:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = []
        bb = []
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = detector(frame, 1)
        print("Number of faces detected: {}".format(len(dets)))
        _, scores, idx = detector.run(frame, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            if d is not None and d.top() >= 0 and d.right() <= gray_image.shape[1] and d.bottom() <= gray_image.shape[0] and d.left() >= 0:
                predicted = predictor(gray_image, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for n in range(len(shape)):
                x = shape[i][n][0]
                y = shape[i][n][1]
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            # the score for each detection.  The score is bigger for more confident detections.
            # The third argument to run is an optional adjustment to the detection threshold,
            # where a negative value will return more detections and a positive value fewer.
            # Also, the idx tells you which of the face sub-detectors matched.  This can be
            # used to broadly identify faces in different orientations.
            print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))

            #SECOND - process the image, rotate, crop, increase contrast, remove noise
            for i in range(0,len(shape)):
                face = []
                rotated_img, landmarks = rotate(gray_image, shape[i])
                cropped_face = crop_face(rotated_img, landmarks)
                eq_face = cv2.equalizeHist(cropped_face)
                filtered_face = cv2.GaussianBlur(eq_face, (5, 5), 0)
                resized_face = cv2.resize(filtered_face, (48, 48), interpolation = cv2.INTER_AREA)
                face.append(resized_face)
                face = np.array(face)
                face = face.reshape(face.shape[0], face.shape[1], face.shape[2], 1)
                face = face.astype('float32') / 255
                prediction = model.predict(face)
                emotion = get_emotion(np.argmax(prediction))
                x = bb[i][0]-10
                y = bb[i][1]+bb[i][2]+20
                cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

        cv2.imshow('Frame', frame)
        cv2.imshow('filtered_face', filtered_face)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


video_capture.release()
cv2.destroyAllWindows()

'''
filtered_face = np.zeros((1, 48, 48, 1))
model = load_model('../../model_lucia_fer.h5')
#model = load_model('../../model_lucia_fer_balanced.h5')
#model = load_model('../../model_lucia_fer_augmentation.h5')
images = []
for filename in os.listdir('images'):
    images.append('images/'+filename)
        
for img in images:
    print(img)
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = []
    bb = []
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(image, 1)
    print("Number of faces detected: {}".format(len(dets)))
    _, scores, idx = detector.run(image, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        if d is not None and d.top() >= 0 and d.right() <= gray_image.shape[1] and d.bottom() <= gray_image.shape[0] and d.left() >= 0:
            predicted = predictor(gray_image, d)
            shape.append(shape_to_np(predicted))
            (x, y, w, h) = rect_to_bb(d)
            bb.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for n in range(0, 68):
            x = shape[i][n][0]
            y = shape[i][n][1]
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        # the score for each detection.  The score is bigger for more confident detections.
        # The third argument to run is an optional adjustment to the detection threshold,
        # where a negative value will return more detections and a positive value fewer.
        # Also, the idx tells you which of the face sub-detectors matched.  This can be
        # used to broadly identify faces in different orientations.
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))

        #SECOND - process the image, rotate, crop, increase contrast, remove noise
        for i in range(0,len(shape)):
            face = []
            #Stage 0: Raw Set
            #Stage 1: Rotation Correction Set
            rotated_img, landmarks = rotate(gray_image, shape[i])
            #Stage 2: Cropped Set
            cropped_face = crop_face(rotated_img, landmarks)
            #Stage 3: Intensity Normalization Set
            image_norm = cv2.normalize(cropped_face, None, 0, 255, cv2.NORM_MINMAX)
            #Stage 4: Histogram Equalization Set
            eq_face = cv2.equalizeHist(image_norm)
            #Stage 5: Smoothed Set
            filtered_face = cv2.GaussianBlur(eq_face, (5, 5), 0)
            resized_face = cv2.resize(filtered_face, (48, 48), interpolation = cv2.INTER_AREA)
            face.append(resized_face)
            face = np.array(face)
            face = face.reshape(face.shape[0], face.shape[1], face.shape[2], 1)
            face = face.astype('float32') / 255
            prediction = model.predict(face)
            emotion = get_emotion(np.argmax(prediction))
            x = bb[i][0]-10
            y = bb[i][1]+bb[i][2]+20
            cv2.putText(image, emotion, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

    cv2.imshow('Frame', image)
    cv2.imshow('filtered_face', filtered_face)
    cv2.waitKey(0)

    

cv2.destroyAllWindows()

