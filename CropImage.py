#import required libraries
import cv2

# read the input image
img = cv2.imread('/Users/ninam/Desktop/Courses/Courses/Embedded&Networking/Embedded yr3/face_recognition/Person/jpeg')

# load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# detect faces in the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# check if any face is detected
if len(faces) > 0:
    # get the first face
    (x, y, w, h) = faces[0]

    # crop the face region from the image
    face_crop = img[y:y+h, x:x+w]

    # display the cropped face
    cv2.imshow('face', face_crop)
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

else:
    print('No face detected.')