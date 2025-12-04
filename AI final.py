import cv2

# Defining the image:
image = cv2.imread("C:\\Users\\lthompson830\\Documents\\shapes.jpg")

# This converts it to a grey image because I don't want it to just memorize color
g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# This makes a color threshold, which means it will scan every pixel and either makes them black (value: 0) or white (value: 1)
_, thresh_image = cv2.threshold(g_image, 220, 255, cv2.THRESH_BINARY)

# Sense all of the shapes are on the same image, these few lines will pick out the coordinates of each shape
# In other words the contour variable will make coordinates to find each shape on the image and remove everything that is unnecessary
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    if i == 0:
        continue

    # These two lines makes the A.I. determine a shape by approximating it
    # #it'll need to use this if the shape is distorted
    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour,epsilon, True)