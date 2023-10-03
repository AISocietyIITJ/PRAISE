# %%
import os
import numpy as np
import cv2

# %%


# %%


# Load the image
image = cv2.imread(pathfile)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Threshold the image
#thresholded_image = cv2.adaptiveThreshold(blurred_image,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,30,3)[1]
thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
# Save the processed image
cv2.imwrite('processed_image.png', thresholded_image)

# Display the preprocessed image
cv2.imshow('Processed Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
''''data_path = r'/home/pranjal/Desktop/code1010/IMG_train.JPG'
categories = os.listdir(data_path)
print (categories)'''

