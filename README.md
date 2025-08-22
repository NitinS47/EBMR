# 🎶 Emotion-Based Music Recommendation System  

An **AI-powered music recommendation system** that detects user emotions from facial expressions and recommends personalized songs accordingly. The system uses **Haar Cascade Classifier** for face detection, an **emotion classification model** to identify emotions (happy, sad, angry, neutral, etc.), and then queries **Gemini** (Google’s generative AI model) to recommend songs that best match the detected emotion.  

---

## 🚀 Features
- **Real-Time Face Detection** – Uses OpenCV’s Haar Cascade for efficient face detection.  
- **Emotion Recognition** – Classifies user emotions from detected faces.  
- **AI-Powered Recommendations** – Sends detected emotion to Gemini API for music suggestions.  
- **Personalized Experience** – Recommends mood-based playlists dynamically.  
- **Scalable Integration** – Can connect with Spotify API or YouTube Music for playback.  

---

## 🛠️ Tech Stack
- **Face Detection**: OpenCV (Haar Cascade)  
- **Emotion Recognition**: Pre-trained deep learning model (CNN / FER library)  
- **Recommendation Engine**: Gemini API (Google Generative AI)  
- **Backend**: Python (Flask / FastAPI)  
- **Frontend (Optional)**: React / Streamlit for UI  
- **Music API**: Spotify / YouTube Music (optional integration)  

---

## 📂 Project Structure
emotion-music-recommender/
│── models/
│ ├── haarcascade_frontalface_default.xml # Haar Cascade model
│ ├── emotion_model.h5 # Trained emotion recognition model
│
│── backend/
│ ├── app.py # API entry point
│ ├── detect_emotion.py # Face + emotion detection
│ ├── recommend.py # Sends emotion to Gemini and retrieves songs
│
│── frontend/ # Optional React/Streamlit UI
│
│── data/ # Sample emotion images
│
│── requirements.txt # Python dependencies
│── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/emotion-music-recommender.git
cd emotion-music-recommender
Create a virtual environment & install dependencies

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

pip install -r requirements.txt
Set up environment variables
Create a .env file in the root directory:

env
Copy
Edit
GEMINI_API_KEY=your_gemini_api_key_here
Run the backend

bash
Copy
Edit
cd backend
python app.py
(Optional) Run the frontend

bash
Copy
Edit
cd frontend
npm install
npm run dev
🔎 Usage
1. Detect Emotion
System captures a frame from the webcam and detects faces using Haar Cascade:

python
Copy
Edit
from detect_emotion import get_emotion

emotion = get_emotion("sample_face.jpg")
print(emotion)  # e.g., "happy"
2. Get Recommendations
Send detected emotion to Gemini for playlist generation:

python
Copy
Edit
from recommend import get_music_recommendations

songs = get_music_recommendations("happy")
print(songs)
Example output:

json
Copy
Edit
[
  {
    "title": "Happy - Pharrell Williams",
    "artist": "Pharrell Williams",
    "url": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH"
  },
  {
    "title": "Uptown Funk",
    "artist": "Mark Ronson ft. Bruno Mars",
    "url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS"
  }
]
📊 Roadmap
 Add direct Spotify/YouTube playback integration

 Support multi-face detection (group emotion)

 Improve emotion classification with CNN/Transformer-based models

 Mobile app integration

🤝 Contributing
Contributions are welcome! Please open issues and pull requests to improve the system.

📜 License
MIT License © 2025 Emotion-Based Music Recommendation System Contributors

yaml
Copy
Edit

