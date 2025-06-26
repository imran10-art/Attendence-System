import cv2
import face_recognition
import pickle
import os

# Load all known face encodings
known_encodings = []
known_names = []

for file in os.listdir("encodings"):
    if file.endswith(".pkl"):
        with open(f'encodings/{file}', 'rb') as f:
            encoding = pickle.load(f)
            known_encodings.append(encoding)
            known_names.append(file.split(".")[0])

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate distances to all known encodings
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        if len(distances) > 0:
            min_distance = min(distances)
            min_distance_index = distances.tolist().index(min_distance)
            # Set a threshold for recognition (lower is stricter, typical is 0.6)
            threshold = 0.5
            if min_distance < threshold:
                name = known_names[min_distance_index]

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()