from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import sqlite3
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'a_really_secret_key_that_is_random_and_long'

# Load dataset for music recommendations
df = pd.read_csv("Dataset.csv", encoding="utf-8")
df.columns = df.columns.str.strip()  # Remove extra spaces from column names

# Ensure dataset contains required columns
expected_columns = {"Emotion", "Weather", "Link"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Missing expected columns: {expected_columns - set(df.columns)}")

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database (if not exists)
def init_db():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT UNIQUE NOT NULL,
                          password TEXT NOT NULL)''')
        conn.commit()

# Function to get weather based on latitude & longitude
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

# Function to get song recommendations
def get_recommendations(emotion, weather):
    filtered_songs = df[
        (df["Emotion"].str.lower().str.strip() == emotion.lower()) &
        (df["Weather"].str.lower().str.strip() == weather.lower())
    ]

    selected_songs = (
        filtered_songs["Link"]
        .dropna()
        .astype(str)
        .str.strip()
        .sample(n=min(3, len(filtered_songs)))
        .tolist()
    )

    return [song.replace("open.spotify.com", "open.spotify.com/embed") for song in selected_songs]

# Route: Home -> Redirect to Login
@app.route("/")
def home():
    return redirect(url_for("login"))

# Route: Music Recommendation Page
@app.route("/index", methods=["GET", "POST"])
def index():
    if "username" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))

    songs = []
    detected_weather = session.get("weather")  # Get weather stored in session

    if request.method == "POST":
        emotion = request.form.get("emotion")
        if detected_weather:
            songs = get_recommendations(emotion, detected_weather)
        else:
            flash("Could not determine weather. Try again!", "warning")

    return render_template("index.html", songs=songs, detected_weather=detected_weather)

# Route: Fetch Weather via API Call (Used by JS)
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

# Route: User Registration
# Route: User Registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]  # Get confirm password field
        
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                flash("Username already exists! Choose a different one.", "warning")
                return redirect(url_for("register"))

            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()

        flash("User registered successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Route: User Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):
            session["username"] = username
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")

# Route: User Logout
@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("weather", None)  # Clear weather on logout
    return redirect(url_for("login"))

# Initialize database on startup
init_db()

if __name__ == "__main__":
    app.run(debug=True)
