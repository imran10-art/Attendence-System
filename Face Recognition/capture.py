import cv2
import face_recognition
import pickle
import os

# Create directory to save encodings if not exists
os.makedirs("encodings", exist_ok=True)

# Input the name of the person
person_name = input("Enter the name of the person: ")

# Start the webcam
cap = cv2.VideoCapture(0)
print("Press 's' to scan and save face, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Scan Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            with open(f'encodings/{person_name}.pkl', 'wb') as f:
                pickle.dump(face_encoding, f)

            print(f"{person_name}'s face encoding saved.")
            break
        else:
            print("No face found. Try again.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()