import os.path
import cv2 as cv
from pathlib import Path

# global variables
img_path = "/Users/zklapman/Desktop/FaceRecognition/images/"
og_img = "webcam_frame_original.png"
face_classifier = cv.CascadeClassifier("/Users/zklapman/Desktop/FaceRecognition/.venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def main():
    # setup webcam
    print("Current directory: " + os.getcwd())
    webcam_index = 0
    window = "Display Window"
    window_width = 0
    window_height = 0
    webcam = cv.VideoCapture(webcam_index)
    if not webcam.isOpened():
        raise IOError("Cannot open webcam")

    # open webcam
    print("Take a selfie!")
    while True:
        ret, video_frame = webcam.read()
        video_frame = cv.resize(video_frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imshow(window, video_frame)
        window_width = cv.getWindowImageRect(window)[2]
        window_height = cv.getWindowImageRect(window)[3]

        k = cv.waitKey(1)
        if k == 27:     # esc pressed - close webcam
            break
        elif k%256 == 32:   # space pressed - take pic and save
            cv.imwrite(os.path.join(img_path, og_img), video_frame)
            print("{} written!".format(og_img))
        elif k ==13:    # enter pressed - open original image and modify
            # read and show original image
            img = cv.imread(os.path.join(img_path, og_img))
            cv.imshow("Original Image", img)
            cv.moveWindow("Original Image", 100,500)

            # detect face
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)    
                    # scale var, scaleFactor, minNeighbors
                    # returns 4 values: x-coord, y-coord, width, height for detected feature of the face
            if faces is ():
                print("No faces found")
            for(x,y,w,h) in faces:
                # TODO: call hat overlay function here and show the bday hat on head

                # show rectangle for face detection
                cv.rectangle(img, (x,y), (x+w, y+h), (127,0,255), 2)
                cv.imshow("Face Detection", img)
                cv.moveWindow("Face Detection", 800,500)
                cv.waitKey(0)

    # close webcam
    cv.destroyAllWindows()

main()