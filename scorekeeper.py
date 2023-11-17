import cv2
import sys

def playVideo():
    file = sys.argv[1]
    cap = cv2.VideoCapture(file)

    cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Player", 1920, 1080)
    
    while(cap.isOpened()):
        success, frame = cap.read()
        if success:
            cv2.imshow('Video Player', frame)
            cv2.waitKey(1000//120)
            closeButton = cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) < 1
            if closeButton: 
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

playVideo()