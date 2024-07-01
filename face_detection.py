# you can use this code to basically detect a face from any image in an instant! Try it out.

# [Viola-Jones Object Detection Framework - 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
# !pip install -q opencv-python
# opencv is required


from google.colab.patches import cv2_imshow
import cv2 as cv
import requests

url = "path_or_url/of/image/from/web_or_uploads"
response = requests.get(url)
with open('image.jpg', 'wb') as file:
    file.write(response.content)

# read image
original_image = cv.imread('../image.jpg')
# convert color image into grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# Load the classifier and create a cascade object for face detection
# Download this from OpenCV repo directly and use it, its easier.
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
face_cascade = cv.CascadeClassifier('/content/haarcascade_frontalface_alt.xml')

# use multi scale detection object to identify the faces
detected_faces = face_cascade.detectMultiScale(grayscale_image)

# let's display the findings
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),  # top-left coordinates
        (column+width, row+height), # bottom-right coordinates
        (0, 255, 0), # rectangle color
        2  # rectangle thickness
    )

# display the image
# cv.imshow('Image', original_image)  # does not work in colab
cv2_imshow(original_image)
# not needed in colab/jupyter actually but required in scripts
# cv.waitKey(0)  # to prevent it from closing immediately
# cv.destroyAllWindows()  # close all the open windows

