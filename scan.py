# Imports
import numpy as np
import argparse
import cv2
import imutils
from skimage.filters.thresholding import threshold_local
from transform import four_point_transform

# Construct an argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())

# Step 1: Edge Detection
image = cv2.imread(args["image"])
orig = image.copy()
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, threshold1=75, threshold2=200)
cv2.imshow("Original Image", image)
cv2.imshow("Edge Detected Image", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Find Contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outlined Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply the four point transform to the image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Give 'black and white' paper effect to the image
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method='gaussian')
warped = (warped > T).astype("uint8") * 255

# Show the Original and Scanned Image
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
