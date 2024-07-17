import os
import cv2

#img = cv2.imread(os.path.join('.','Data','bird.jpg'))
img = cv2.imread(os.path.join('D:\ML\Data','bird.jpg'))
print(img.shape)

cropped_img = img[55:380,150:625] #(height, width)

cv2.imshow('img',cropped_img)
cv2.waitKey(0)