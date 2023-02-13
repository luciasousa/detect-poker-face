import cv2

alpha = 1.5 # Contrast control
beta = 5 # Brightness control

image = cv2.imread('example_fer.jpg')
cv2.imshow('image original', image)
#convert to gray
input_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#change intensity of image
input_img = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)

#zoom in image by 
input_img = cv2.resize(input_img, (0,0), fx=3, fy=3)
#resize image

#show image
cv2.imshow('image', input_img)
cv2.waitKey(0)