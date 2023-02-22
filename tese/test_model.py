#capture video of a person and detect neutral face

from tensorflow.keras.models import load_model
import cv2
import dlib
import numpy as np

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	
	return (x, y, w, h)

# Convert the facial landmarks dlib format to numpy
def shape_to_np(shape):
	# pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates
	# that map to facial structures on the face
	landmarks = np.zeros((68,2), dtype = int)

	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)

	return landmarks

def to_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
		#for (x, y) in shape:
		#	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	return shape, bb

def get_emotion(label):
	if label == 0:
		return "Neutral"
	elif label == 1:
		return "Not Neutral"


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('../../model_lucia_ck.h5')
model.summary()

while(True):
    ret, frame = cap.read()
    if ret == True:
        # Detect faces
        gray_image = to_grayscale(frame)
        shape, bb = detect_face(frame, gray_image)
	
        ###TODO: 
        #frame mais alinhada com a camara
        #pré-processamento
        #calcular tempo de inferencia
        ###

        # predict emotion
        if len(bb) > 0:
            for i in range(len(bb)):
                x, y, w, h = bb[i]
                roi = gray_image[y:y+h, x:x+w]
                roi = cv2.resize(roi, (96,96))
                roi = roi.astype('float') / 255.0
                roi = np.array(roi, dtype=np.float32)
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, -1)
                prediction = model.predict(roi)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, get_emotion(maxindex), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
