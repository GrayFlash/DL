import cv2 as cv
class haar:
    def Region_OI(image):
        # Takes image as input and returns biggest face detected
        # returns starting point (x, y) and height, width (h, w) of face
        face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)
        max_area = 0
        x_1=0
        y_1=0
        w_1=0
        h_1=0
        for face in faces:
            x, y, w, h = faces[0]
            if w*h > max_area:
                max_area = w*h
                x_1, y_1, w_1, h_1 = face
        
        return x_1,y_1,w_1,h_1