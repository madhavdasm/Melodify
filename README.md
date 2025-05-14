

# 🎵 Melodify - AI-based Indian Music Recommendation System

**Melodify** is a Flask-based web application that recommends Indian classical and film songs to users based on their detected **emotion** (via facial expression) and **current weather** (via geolocation). The system uses AI for face emotion detection and hybrid recommendation logic powered by user feedback and content similarity.

---

## 🚀 Features

* **Emotion Detection** using `DeepFace`
* **Weather Detection** via OpenWeatherMap API
* **Hybrid Song Recommendation** based on:

  * Emotion + Weather match
  * User feedback (likes)
  * Cosine similarity of singer-composer combinations
* **Like/Dislike Feedback System** to personalize future recommendations
* **Avoids repeating recently recommended songs**
* **Similarity Matrix Visualization** (optional)
* **User Registration & Login** with hashed passwords
* Backend in **Flask**, frontend via HTML templates

---

## 🧠 How It Works

1. **User logs in** or creates an account.
2. **Face image** is captured and analyzed via DeepFace to detect emotion.
3. **Weather** is detected using the user's geolocation.
4. **A song is recommended** from a curated dataset, based on:

   * Matching EMOTION + WEATHER
   * Prior liked songs (cosine similarity on artist data)
   * Fallback to random if needed
5. **User can like/dislike the song**, improving future recommendations.
6. Feedback is stored in **SQLite** and used to adjust future suggestions.

---

## 📁 Dataset Structure

The dataset (`Dataset.csv`) must include the following columns:

* `EMOTION`
* `WEATHER`
* `LINK` (YouTube or audio URL)
* `RAGA`
* `SINGER`
* `COMPOSER`

---

## 🛠️ Tech Stack

* **Frontend**: HTML, Bootstrap (in `templates/`)
* **Backend**: Python, Flask
* **AI Libraries**:

  * `DeepFace` for facial emotion recognition
  * `scikit-learn` for TF-IDF and cosine similarity
* **Database**: SQLite (`users.db`)
* **Data Visualization**: Seaborn + Matplotlib (for similarity matrix)

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/Melodify.git
cd Melodify
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

(Ensure `deepface`, `opencv-python`, `flask`, `requests`, `scikit-learn`, `matplotlib`, `seaborn`, etc. are included in `requirements.txt`.)

### 3. Add Dataset

Place `Dataset.csv` in the root directory with appropriate columns.

### 4. Configure Weather API

Replace the OpenWeatherMap API key in `get_weather()` function with your own:

```python
API_KEY = "your_openweathermap_api_key"
```

### 5. Run the App

```bash
python app.py
```


---

## 🔐 User Auth

* Passwords are securely hashed using `werkzeug.security`.
* SQLite stores users and feedback in `users.db`.
* Table structures are auto-created on app run (`init_db()` function).

---

## 🎯 Future Enhancements

* 🎤 Voice-based emotion recognition
* 📱 Mobile-friendly UI
* 🎶 Spotify/Youtube integration for song playback
* 💾 Admin dashboard for adding/removing songs
* 🤖 Reinforcement learning for smarter personalization

---

## 🤝 Contributing

Feel free to fork and submit pull requests or suggestions!

---

## 📷 Screenshots


---

## 📄 License

This project is open-source and free to use for educational purposes.

---

Let me know if you’d like me to generate a `requirements.txt` file or provide example screenshots section too.
