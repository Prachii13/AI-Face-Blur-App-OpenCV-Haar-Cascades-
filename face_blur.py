import cv2
import os

CASCADE = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE)

def blur_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y+h, x:x+w] = blurred
    return image

def process_webcam():
    cap = cv2.VideoCapture(0)
    print("📷 Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = blur_faces(frame)
        cv2.imshow("AI Face Blur", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    image = cv2.imread(image_path)
    blurred = blur_faces(image)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === Use one of the following ===
# process_webcam()
# process_image('static/uploads/sample.jpg')
