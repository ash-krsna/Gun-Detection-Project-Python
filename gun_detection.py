import cv2
import imutils
import datetime

# Load cascades
gun_cascade = cv2.CascadeClassifier('cascade.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if gun_cascade.empty() or face_cascade.empty():
    print("Error loading cascade files")
    exit()

camera = cv2.VideoCapture(0)

gun_exist = False

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    guns = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw faces (GREEN = HUMAN SAFE)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(
            frame,
            (fx, fy),
            (fx + fw, fy + fh),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "HUMAN",
            (fx, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw guns (RED)
    for (gx, gy, gw, gh) in guns:
        gun_exist = True

        cv2.rectangle(
            frame,
            (gx, gy),
            (gx + gw, gy + gh),
            (0, 0, 255),
            3
        )
        cv2.putText(
            frame,
            "GUN",
            (gx, gy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Security Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
