import cv2

eye_ref = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)

def eye_detection(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_ref.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=3, minSize=(100,100))
    return eyes

def frame_box(frame):

    for x,y,w,h in eye_detection(frame):
        cv2.rectangle(frame,(x,y),(x + w, y + h), (0, 0, 0), 4)

def main():
    
    while True:
        result, frame = camera.read()
        frame_box(frame)
        cv2.imshow("Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()