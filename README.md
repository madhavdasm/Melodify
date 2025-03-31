# Music Recommendation System Based on Emotion and Weather

## Overview

This is a Flask-based web application that recommends music based on:
- User's current emotional state (detected via facial recognition)
- Local weather conditions
- User's past feedback on recommendations

The system uses a hybrid recommendation approach combining:
- Content-based filtering (cosine similarity on song metadata)
- Context-aware filtering (emotion + weather matching)
- Random selection as fallback for new users

## Key Features

1. **User Authentication**
   - Secure login/registration with password hashing
   - Session management

2. **Emotion Detection**
   - Uses DeepFace library for facial emotion analysis
   - Maps detected emotions to musical moods
   - Handles edge cases (no face detected, low confidence)

3. **Weather Integration**
   - Connects to OpenWeatherMap API
   - Classifies weather into Sunny/Rainy/Cloudy
   - Stores weather context in session

4. **Hybrid Recommendation Engine**
   - For new users: Uses emotion + weather filtering
   - For existing users: Content-based filtering on liked songs' metadata
   - Avoids repeating recent recommendations
   - Tracks user feedback (likes/dislikes) to improve future suggestions

5. **Visual Analytics**
   - Generates similarity heatmaps for recommendations
   - Logs detailed emotion detection probabilities

## Technical Stack

- **Backend**: Python (Flask)
- **Database**: SQLite
- **Computer Vision**: OpenCV, DeepFace
- **Machine Learning**: scikit-learn (TF-IDF, cosine similarity)
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML/CSS/JavaScript

## Setup Instructions

1. **Prerequisites**
   - Python 3.7+
   - pip

2. **Installation**
   ```bash
   git clone [repository-url]
   cd [repository-folder]
   pip install -r requirements.txt
   ```

3. **Configuration**
   - Get an OpenWeatherMap API key
   - Create a `.env` file in the root directory and add:
     ```env
     WEATHER_API_KEY=your_api_key_here
     ```
   - Ensure `Dataset.csv` is in the project root

4. **Initialize Database**
   ```bash
   python
   >>> from app import init_db
   >>> init_db()
   >>> exit()
   ```

5. **Run Application**
   ```bash
   python app.py
   ```

## Usage

1. Register a new account or login
2. Allow camera access for emotion detection
3. Grant location access for weather data
4. Receive personalized music recommendations
5. Provide feedback (like/dislike) to improve future recommendations

## File Structure

```
├── app.py                # Main application file
├── Dataset.csv           # Music dataset
├── users.db              # SQLite database (created after first run)
├── static/               # Static files
│   └── captured_image.png # Temporary image storage
├── templates/            # HTML templates
│   ├── face.html         # Emotion detection interface
│   ├── index.html        # Recommendation interface
│   ├── login.html        # Login page
│   └── register.html     # Registration page
├── .env                  # Environment variables (API key)
└── requirements.txt      # Python dependencies
```

## Dataset Format

The system expects `Dataset.csv` with these required columns:

| EMOTION  | WEATHER | LINK | RAGA | SINGER | COMPOSER |
|----------|--------|------|------|--------|----------|
| Happy    | Sunny  | [YouTube link] | Raag Bhairav | Arijit Singh | A.R. Rahman |
| Sad      | Rainy  | [YouTube link] | Raag Yaman | Lata Mangeshkar | R.D. Burman |
| Neutral  | Cloudy | [YouTube link] | Raag Darbari | Kishore Kumar | S.D. Burman |

## API Endpoints

- `/register` - User registration
- `/login` - User authentication
- `/face` - Emotion detection interface
- `/capture` - Handles image capture
- `/detect_emotion` - Processes emotion analysis
- `/weather` - Fetches weather data
- `/index` - Main recommendation interface
- `/feedback` - Processes user feedback

## Troubleshooting

1. **Emotion detection fails**
   - Ensure good lighting
   - Face should be clearly visible
   - Check browser console for errors

2. **Weather not loading**
   - Verify API key is valid
   - Check internet connection
   - Ensure location permissions are granted

3. **No recommendations**
   - Verify Dataset.csv is properly formatted
   - Check application logs for errors

## License

This project is open-source under the MIT License.

