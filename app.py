from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import sqlite3
import pandas as pd
import base64
import cv2
import numpy as np
from deepface import DeepFace
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import random
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'a_really_secret_key_that_is_random_and_long'

# Load Dataset
df = pd.read_csv("Dataset.csv", encoding="utf-8").applymap(lambda x: x.strip() if isinstance(x, str) else x)
expected_columns = {"EMOTION", "WEATHER", "LINK", "RAGA"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Missing expected columns: {expected_columns - set(df.columns)}")

# Prepare TF-IDF vectorizer on RAGA column
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["RAGA"].astype(str))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Database Connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL)''')
        conn.commit()

# Weather API
@app.route("/weather", methods=["POST"])
def weather():
    data = request.json
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat and lon:
        weather_condition = get_weather(lat, lon)
        if weather_condition:
            session["weather"] = weather_condition  # Store in session
            return jsonify({"weather": weather_condition})
    
    return jsonify({"error": "Could not determine weather"}), 400

def get_weather(lat, lon):
    API_KEY = "8c7c5c5ee0cac54b1ea472e5ea37de11"  # Replace with your OpenWeatherMap API key
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

    url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        weather_code = data["weather"][0]["id"]
        wind_speed = data["wind"]["speed"]

        # Determine weather condition
        if 200 <= weather_code < 600:
            weather_condition = "Rainy"
        elif 700 <= weather_code < 800:
            weather_condition = "Cloudy"
        elif weather_code == 800:
            weather_condition = "Sunny"
        else:
            weather_condition = "Cloudy"

        # Detect Windy conditions if wind speed > 6 m/s
        if wind_speed > 6:
            weather_condition = "Windy"

        return weather_condition
    else:
        return None  # Return None if weather data isn't available

def get_recommended_song():
    """Recommend a song avoiding repetition using cosine similarity."""
    try:
        emotion = session.get("emotion", "").strip().lower()
        weather = session.get("weather", "").strip().lower()
        prev_songs = session.get("recommended_songs", [])

        logging.debug(f"Fetching recommendation for Emotion: {emotion}, Weather: {weather}")

        if not emotion or not weather:
            return None

        # Filter dataset based on emotion and weather
        filtered_songs = df[
            (df["EMOTION"].str.strip().str.lower() == emotion) & 
            (df["WEATHER"].str.strip().str.lower() == weather)
        ]

        if filtered_songs.empty:
            return None

        # Exclude previously recommended songs
        available_songs = filtered_songs[~filtered_songs["LINK"].isin(prev_songs)]

        if available_songs.empty:
            logging.warning("All songs have been recommended. Resetting history.")
            session["recommended_songs"] = []  # Reset history
            available_songs = filtered_songs  # Recommend from all

        # Select a song randomly
        selected_song = available_songs.sample(n=1).iloc[0]

        # Update session with recommended song
        session["raga"] = selected_song["RAGA"]
        session.setdefault("recommended_songs", []).append(selected_song["LINK"])

        return selected_song["LINK"]

    except Exception as e:
        logging.error("Error filtering songs: " + str(e))
        return None

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "emotion" not in session or "weather" not in session:
            return jsonify({"error": "Emotion or weather data missing"}), 400
        
        song_link = get_recommended_song()
        session["song"] = song_link  # Store in session

        if song_link:
            return jsonify({
                "song": song_link, 
                "raga": session.get("raga", ""),  # Send updated RAGA
                "emotion": session.get("emotion", "").capitalize(),
                "weather": session.get("weather", "").capitalize()
            })
        else:
            return jsonify({"error": "No recommendations found"}), 404
    
    # Clear the song if coming via GET request (fresh page load)
    session.pop("song", None)
    return render_template("index.html", song=None)


@app.route("/face")
def face():
    if "username" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))
    return render_template("face.html")

@app.route("/capture", methods=["POST"])
def capture():
    try:
        image_data = request.json["image"].split(",")[1]
        image_np = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        cv2.imwrite("static/captured_image.png", cv2.imdecode(image_np, cv2.IMREAD_COLOR))
        return jsonify({"image_path": "/static/captured_image.png"})
    except Exception as e:
        return jsonify({"error": str(e)})

import logging

logging.basicConfig(filename="emotion_debug.log", level=logging.INFO)

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        result = DeepFace.analyze("static/captured_image.png", actions=['emotion'], enforce_detection=False)

        if not result or "dominant_emotion" not in result[0]:
            return jsonify({"error": "No face detected"})

        emotions = result[0]["emotion"]
        
        # Log emotion probabilities for debugging
        logging.info(f"Emotion Probabilities: {emotions}")

        # Get the most probable emotion
        most_probable_emotion = max(emotions, key=lambda k: float(emotions[k]))

        emotion_mapping = {
            "happy": "Happy",
            "sad": "Sad",
            "angry": "Angry",
            "neutral": "Neutral",
            "fear": "Sad",
            "disgust": "Angry",
            "surprise": "Happy",
        }

        # Assign mapped emotion based on highest probability
        mapped_emotion = emotion_mapping.get(most_probable_emotion, "Neutral")

        session["emotion"] = mapped_emotion

        # Log final detected emotion
        logging.info(f"Detected Emotion: {mapped_emotion}")

        return jsonify({
            "emotion": mapped_emotion,
            "redirect": url_for("weather")
        })

    except Exception as e:
        logging.error("Error detecting emotion: " + str(e))
        return jsonify({"error": str(e)}), 500




    
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username, password, confirm_password = request.form["username"], request.form["password"], request.form["confirm_password"]
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))
        
        with get_db_connection() as conn:
            if conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
                flash("Username already exists!", "warning")
                return redirect(url_for("register"))
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, generate_password_hash(password)))
            conn.commit()
        flash("User registered successfully! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username, password = request.form["username"], request.form["password"]
        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["username"] = username
            return redirect(url_for("face"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

init_db()

if __name__ == "__main__":
    app.run(debug=True)
