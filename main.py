import cv2
import numpy as np
import joblib

# Load ML models
face_model = joblib.load("face_shape_model.pkl")
skin_model = joblib.load("skin_model.pkl")
hair_model = joblib.load("hair_model.pkl")

# Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Multi-face warning - FIXED: Draw at top center with better visibility
    if len(faces) > 1:
        # Draw red rectangle around the entire frame as warning
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
        
        # Add warning text with better positioning and styling
        text = "MULTIPLE FACES DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 60
        
        # Draw text with outline for better visibility
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)  # Black outline
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # Red text

    # Process only first face
    y_offset = 60  # Starting Y position for text
    for (x, y, w, h) in faces[:1]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        ratio = h / w

        shape = face_model.predict([[ratio]])[0]
        tone = skin_model.predict([[np.mean(gray[y:y+h, x:x+w])]])[0]
        hair = hair_model.predict([[shape]])[0]

        shapes = ["Oval","Round","Square"]
        tones = ["Fair","Medium","Dark"]
        hairs = ["Any","Side Bangs","Curly"]

        # Draw face analysis results with better positioning
        cv2.putText(frame, "Face Analysis:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Shape: {shapes[int(shape)]}", (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Skin: {tones[int(tone)]}", (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Hair: {hairs[int(hair)]}", (10, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Instructions
    cv2.putText(frame, "Press ESC to exit", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Smart ML Face System", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()