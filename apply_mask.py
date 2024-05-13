import cv2
import numpy as np


# Read the first image (the black region mask)
black_region_mask = cv2.imread('segmented_black_region.jpg', cv2.IMREAD_GRAYSCALE)

# Read the second image
second_image = cv2.imread('./out3/00014-00225.png')

# Invert the black region mask
inverse_mask = cv2.bitwise_not(black_region_mask)

# Apply the black region mask on the second image
result = cv2.bitwise_and(second_image, second_image, mask=inverse_mask)

# Save the final result
cv2.imwrite('final_image.jpg', result)




