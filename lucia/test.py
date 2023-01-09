import cv2
import numpy as np

def rotate(image, landmarks):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -30, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (nW, nH))
    rotated_landmarks = np.dot(M, np.vstack([landmarks.T, np.ones(landmarks.shape[0])]))[:2].T
    return rotated, rotated_landmarks

img = cv2.imread('test.jpg')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#rotated_img, landmarks = rotate(gray_image, shape[i])
#cropped_face = crop_face(rotated_img, landmarks)

image_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
eq_face = cv2.equalizeHist(image_norm)
#filtered_face = cv2.GaussianBlur(eq_face, (5, 5), 0)

cv2.imshow('image', img)
cv2.imshow('filtered', eq_face)

cv2.waitKey(0)