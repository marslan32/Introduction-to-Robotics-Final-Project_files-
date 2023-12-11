import cv2
import numpy as np

from camera_class import IntelCam

camera = IntelCam()
camera.initialize()
mask_path = camera.capture_image()
camera.close()

image = cv2.imread('images/rgb.jpeg')  # Replace 'checkerboard.jpg' with the path to your image
# image = cv2.imread("chessboard_10x8_25mm.jpg")  # Replace 'checkerboard.jpg' with the path to your image

# Find the dimensions of the image
height, width, _ = image.shape

# Calculate the center point
center_x = width // 2
center_y = height // 2

# Draw a marker at the center of the image
marker_radius = 25
cv2.circle(image, (center_x, center_y), marker_radius, (0, 0, 255), -1)  # -1 fills the circle

# Define the size of the new markers in millimeters
square_size_mm = 25

# Calculate the size of the new markers in pixels based on the image's resolution
# You'll need to know the DPI (dots per inch) of the image to convert mm to pixels accurately
# For example, if the DPI is 300, then 1 mm = 300/25.4 pixels
dpi = 96  # Replace with the DPI of your image
square_size_pixels = int(square_size_mm * dpi / 25.4)

# Calculate the coordinates for the top-right corner of the square
square_top_left_x = center_x
square_top_left_y = center_y - square_size_pixels

# Draw the square
square_color = (0, 255, 0)  # Green color for the square
square_thickness = 2  # Thickness of the square's border
cv2.rectangle(image, (square_top_left_x, square_top_left_y), (center_x + square_size_pixels, center_y), square_color, square_thickness)

# Save or display the final image with the square

cv2.imwrite('marked_checkerboard.jpg', image)  # Save the marked image
