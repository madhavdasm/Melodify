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
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'a_really_secret_key_that_is_random_and_long'

# Load Dataset
df = pd.read_csv("Dataset.csv", encoding="utf-8").applymap(lambda x: x.strip() if isinstance(x, str) else x)
expected_columns = {"EMOTION", "WEATHER", "LINK", "RAGA"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Missing expected columns: {expected_columns - set(df.columns)}")



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
        conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL,
                          song TEXT NOT NULL,
                          feedback TEXT NOT NULL)''')
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
    

        # Determine weather condition
        if 200 <= weather_code < 600:
            weather_condition = "Rainy"
        elif 700 <= weather_code < 800:
            weather_condition = "Cloudy"
        elif weather_code == 800:
            weather_condition = "Sunny"
        else:
            weather_condition = "Cloudy"


        return weather_condition
    else:
        return None  # Return None if weather data isn't available

def get_recommended_song():
    """Improved hybrid recommendation with better error handling"""
    try:
        emotion = session.get("emotion", "").lower()
        weather = session.get("weather", "").lower()
        username = session.get("username")
        
        if not all([emotion, weather, username]):
            print("!! Missing required session data !!")
            return None

        prev_recommendations = session.get("recommended_songs", [])
        
        # Clear history after 4 recommendations
        if len(prev_recommendations) >= 4:
            prev_recommendations = []
            session["recommended_songs"] = []
            print("\n--- Cleared recommendation history ---")

        # Get all possible candidates first
        candidates = df[
            (df["EMOTION"].str.lower() == emotion) & 
            (df["WEATHER"].str.lower() == weather) &
            (~df["LINK"].isin(prev_recommendations))
        ]
        
        if candidates.empty:
            print("!! No songs match current filters !!")
            return None

        # 1. Try cosine similarity if user has liked songs
        with get_db_connection() as conn:
            liked_count = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE username = ? AND feedback = 'like'",
                (username,)
            ).fetchone()[0]
        
        if liked_count > 0:
            try:
                cosine_recs = get_cosine_recommendations(username, emotion, weather, prev_recommendations)
                if cosine_recs:
                    print(f"\n=== COSINE RECOMMENDATION ===")
                    print(f"Based on {liked_count} liked songs")
                    print(f"Recommended: {cosine_recs[0]}")
                    selected_link = cosine_recs[0]  # Get link only
                    selected_song_data = df[df["LINK"] == selected_link].iloc[0]
                    session["raga"] = selected_song_data["RAGA"]
                    print("cosine song format with link and raga is ",cosine_recs[0],selected_song_data["RAGA"])
                    session.setdefault("recommended_songs", []).append(cosine_recs[0])  # Safe append
                    return cosine_recs[0]
            except Exception as e:
                print(f"Cosine similarity failed: {str(e)}")
                # Fall through to random if cosine fails

        # 2. Fallback to random selection
        random_rec = candidates.sample(n=1).iloc[0]["LINK"]
                # FALLBACK TO RANDOM SELECTION
        selected_link = random_rec  # Get link only

        # Fetch the FULL row from ORIGINAL df (to guarantee correct raga)
        selected_song_data = df[df["LINK"] == selected_link].iloc[0]

        # Store in session
        session["raga"] = selected_song_data["RAGA"]
        print("random song format with link and raga is ",random_rec,selected_song_data["RAGA"])
        print(f"\n=== RANDOM RECOMMENDATION ===")
        print(f"From {len(candidates)} candidates")
        print(f"Recommended: {random_rec}")
        session.setdefault("recommended_songs", []).append(random_rec)
        return random_rec

    except Exception as e:
        print(f"!! Recommendation system error: {str(e)} !!")
        logging.error(f"Recommendation error: {str(e)}")
        return None
    
def plot_similarity_matrix(username, emotion, weather):
    """Generate a heatmap of song similarities for current context"""
    try:
        # Get liked songs
        with get_db_connection() as conn:
            liked_songs = conn.execute(
                "SELECT song FROM feedback WHERE username = ? AND feedback = 'like'",
                (username,)
            ).fetchall()
        liked_songs = [row[0] for row in liked_songs]
        
        # Filter songs for current context
        context_songs = df[
            (df["EMOTION"].str.lower() == emotion.lower()) & 
            (df["WEATHER"].str.lower() == weather.lower())
        ]
        
        if len(context_songs) < 2:
            print("Not enough songs to plot similarity")
            return
            
        # Prepare features
        features = (
            context_songs["SINGER"].str.lower() + " " + 
            context_songs["COMPOSER"].str.lower()
        )
        
        # Vectorize
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(features)
        
        # Calculate similarity
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Prepare labels
        labels = []
        for idx, row in context_songs.iterrows():
            label = f"{row['SINGER']} - {row['COMPOSER']}\n"
            if row['LINK'] in liked_songs:
                label += "(Liked)"
            labels.append(label)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1
        )
        plt.title(f"Song Similarity Matrix\n{emotion.capitalize()} / {weather.capitalize()}")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save and show
        plt.savefig('similarity_matrix.png')
        plt.show()
        print("Similarity plot saved as similarity_matrix.png")
        
    except Exception as e:
        print(f"Error generating similarity plot: {str(e)}")

def clean_artists(text):
    return [a.strip().lower() for a in str(text).split(",")]

df["SINGERS"] = df["SINGER"].apply(clean_artists)
df["COMPOSERS"] = df["COMPOSER"].apply(clean_artists)

# Create a lookup table for fast filtering
emotion_weather_groups = df.groupby(["EMOTION", "WEATHER"]) 

@app.route("/feedback", methods=["POST"])
def feedback():
    if "username" not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.json
    song_name = data.get("song")
    feedback_type = data.get("feedback")

    if not song_name or feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "Invalid feedback data"}), 400

    username = session["username"]
    update_feedback(username, song_name, feedback_type)
    

    
    return "", 204

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
         # Generate similarity visualization (NEW CODE)
        #try:
           # plot_similarity_matrix(
            #    session['username'],
             #   session['emotion'],
              #  session['weather']
            #)
        #except Exception as e:
         #   print(f"Couldn't generate similarity plot: {str(e)}")
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

def update_feedback(username, song_name, feedback):
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT feedback FROM feedback WHERE username = ? AND song = ?",
                (username, song_name))
            existing = cursor.fetchone()

            if existing:
                conn.execute(
                    "UPDATE feedback SET feedback = ? WHERE username = ? AND song = ?",
                    (feedback, username, song_name))
            else:
                conn.execute(
                    "INSERT INTO feedback (username, song, feedback) VALUES (?, ?, ?)",
                    (username, song_name, feedback))
            conn.commit()
            
        logging.info(f"Feedback updated: {username} -> {song_name} -> {feedback}")
        return True
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        return False

def get_cosine_recommendations(username, emotion, weather, prev_recommendations):
    """Cosine similarity recommendations based only on Singer and Composer"""
    try:
        # Get user's liked songs from database
        with get_db_connection() as conn:
            liked_songs = conn.execute(
                "SELECT song FROM feedback WHERE username = ? AND feedback = 'like'", 
                (username,)
            ).fetchall()
        
        if not liked_songs:
            print("No liked songs found for cosine similarity")
            return None
            
        liked_songs = [row[0] for row in liked_songs]
        
        # Filter candidates matching current context
        candidates = df[
            (df["EMOTION"].str.lower() == emotion.lower()) & 
            (df["WEATHER"].str.lower() == weather.lower()) &
            (~df["LINK"].isin(prev_recommendations + liked_songs))
        ]
        
        if candidates.empty:
            print("No candidate songs available")
            return None
        
        # Get liked songs that match current emotion/weather
        liked_songs_df = df[
            (df["LINK"].isin(liked_songs)) &
            (df["EMOTION"].str.lower() == emotion.lower()) &
            (df["WEATHER"].str.lower() == weather.lower())
        ]
        
        if liked_songs_df.empty:
            print("Liked songs don't match current emotion/weather")
            return None
        
        print(f"Found {len(liked_songs_df)} liked songs for current context")
        
        # Create TF-IDF vectors for Singer+Composer
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(","), lowercase=True)
        
        # Combine Singer and Composer with comma separation
        liked_features = liked_songs_df["SINGER"] + "," + liked_songs_df["COMPOSER"]
        candidate_features = candidates["SINGER"] + "," + candidates["COMPOSER"]
        
        # Fit and transform all features
        all_features = pd.concat([liked_features, candidate_features])
        tfidf_matrix = tfidf.fit_transform(all_features)
        
        # Calculate similarity between liked and candidate songs
        n_liked = len(liked_songs_df)
        sim_matrix = cosine_similarity(tfidf_matrix[:n_liked], tfidf_matrix[n_liked:])
        avg_scores = sim_matrix.mean(axis=0)  # Average similarity across all liked songs
        
        if len(avg_scores) == 0:
            print("No similarity scores calculated")
            return None
            
        # Get the most similar song
        top_idx = np.argmax(avg_scores)
        similarity_score = avg_scores[top_idx]
        
        if similarity_score < 0.1:  # Minimum similarity threshold
            print(f"Similarity score too low: {similarity_score:.2f}")
            return None
            
        recommended_song = candidates.iloc[top_idx]["LINK"]
        print(f"Recommended via cosine similarity (score: {similarity_score:.2f}): {recommended_song}")
        print("recommended song in cosine format is",recommended_song)
        return [recommended_song]
        
    except Exception as e:
        print(f"Error in cosine similarity: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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
