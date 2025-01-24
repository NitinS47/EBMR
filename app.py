import os
import cv2
import numpy as np
import streamlit as st
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import google.generativeai as genai
import time

# Configure Google Generative AI
genai.configure(api_key="AIzaSyAnK63UvpP6vu8AtRGDeKa7blhYQ-LX9FE")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load model
model_keras = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model_keras.load_weights('D:\\Projects\\Gen AI Project\\Emotion Based Music Player\\Emotion-based-music-recommendation-system-main\\model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Streamlit UI
st.title("Emotion-Based Music Recommendation")
st.markdown("<h5 style='text-align: center; color: grey;'>Click 'Start Camera' to scan your emotions</h5>", unsafe_allow_html=True)

camera_button = st.button("Start Camera")

if camera_button:
    st.markdown("<h5 style='text-align: center; color: grey;'>Align your face in front of the camera</h5>", unsafe_allow_html=True)
    stframe = st.empty()  # Placeholder for the video stream

    cap = cv2.VideoCapture(0)
    detected_emotions = []

    if not cap.isOpened():
        st.error("Unable to access the webcam.")
    else:
        start_time = time.time()
        while time.time() - start_time < 20:  # Run for 20 seconds
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Detect faces and predict emotions
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_expanded = np.expand_dims(np.expand_dims(roi_resized, -1), 0)
                prediction = model_keras.predict(roi_expanded)
                max_index = int(np.argmax(prediction))
                detected_emotion = emotion_dict[max_index]
                detected_emotions.append(detected_emotion)

                # Draw a rectangle around the face and label the emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the frame in the Streamlit app
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        st.success("Emotion detection completed!")
        cv2.destroyAllWindows()

        # Process detected emotions
        if detected_emotions:
            most_common_emotion = Counter(detected_emotions).most_common(1)[0][0]

            # Use Generative AI to suggest songs for the detected emotion
            st.markdown(f"<h5 style='text-align: center; color: grey;'>Detected Emotion: <b>{most_common_emotion}</b></h5>", unsafe_allow_html=True)
            response = model.generate_content(f"Suggest songs for the emotion {most_common_emotion}. Just give me the names and their respective artists. Do not give me any description or anything. Give me like 10 songs.")

            # Display recommended songs
            st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs</b></h5>", unsafe_allow_html=True)
            st.text(response.text)
else:
    st.markdown("<h5 style='text-align: center; color: grey;'>Click the button to begin!</h5>", unsafe_allow_html=True)