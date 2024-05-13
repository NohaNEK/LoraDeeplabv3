import cv2
import numpy as np

# Read the RGB mask image
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
mask = cv2.imread('/media/fahad/DATA_2/stable_diff/auttomate/stable-diffusion-webui/output/txt2img-images/00014-00225.png')

# Convert the image to grayscale
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to get a binary mask of the black regions
_, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Invert the binary mask
binary_mask = cv2.bitwise_not(binary_mask)

# Find contours of black regions
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
black_segmented = np.zeros_like(mask)
cv2.drawContours(black_segmented, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Show the segmented black region
cv2.imshow('Segmented Black Region', black_segmented)
cv2.waitKey(1000)
cv2.destroyAllWindows()

cv2.imwrite('segmented_black_region.jpg', black_segmented)

