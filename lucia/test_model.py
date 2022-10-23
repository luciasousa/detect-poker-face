from tensorflow.keras.models import load_model
import sys
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
	elif label == 1:
		return "Anger"
	elif label == 2:
		return "Disgust"
	elif label == 3:
		return "Fear"
	elif label == 4:
		return "Happiness"
	elif label == 5:
		return "Sadness"
	elif label == 6:
		return "Surprise"
    
#FIRST - detect the face with dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('model.h5')
model.summary()

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    image = cv2.imread(f)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = []
    bb = []
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    _, scores, idx = detector.run(img, 1, -1)
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
            processed_image, landmarks = rotate(gray_image, shape[i])
            processed_image = crop_face(processed_image, landmarks)
            image = draw_contour(landmarks)
            # Gaussian blurring is highly effective in removing Gaussian noise from an image.
            processed_image = cv2.GaussianBlur(processed_image,(3,3),0)
            # Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.
            processed_image = cv2.equalizeHist(processed_image)
            resize_image = cv2.resize(processed_image, (48,48))
            face.append(resize_image)
            face = np.array(face)
            face = face.reshape(face.shape[0], face.shape[1], face.shape[2], 1)
            face = face.astype('float32') / 255
            #THIRD - infer emotion
            prediction = model.predict(face)
            emotion = get_emotion(np.argmax(prediction))
            x = bb[i][0]-10
            y = bb[i][1]+bb[i][2]+20
            cv2.putText(image, emotion, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

    cv2.imshow("image", image)
    cv2.imshow("processed_image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
