from flask import Flask, render_template, request, redirect, session, url_for, jsonify, send_file
import sqlite3
import os
import platform
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import json
import pyttsx3
import tempfile
import speech_recognition as sr
from gtts import gTTS
import threading
import uuid
import pickle
import numpy as np
nltk.download('punkt_tab')

app = Flask(__name__, static_folder='static')
app.secret_key = '123'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set()
audio_folder = os.path.join(app.static_folder, 'audio')


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)


model = load_model('chatbot_model.h5')
with open('intents.json') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)


wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def init_db():
    with sqlite3.connect("data.db") as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )''')
init_db()


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [wnl.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(msg):
    bow = bag_of_words(msg, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

@app.route('/')
def login():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_user():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    try:
        with sqlite3.connect("data.db") as conn:
            conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
        return redirect(url_for('login'))
    except sqlite3.IntegrityError:
        return "Email already exists!"

@app.route('/login', methods=['POST'])
def login_user():
    email = request.form['email']
    password = request.form['password']
    with sqlite3.connect("data.db") as conn:
        cursor = conn.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
    if user:
        session['user_id'] = user[0]
        session['chat_history'] = []  
        return redirect(url_for('bot'))
    else:
        return "Invalid login credentials"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/bot')
def bot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    chat_history = session.get('chat_history', [])
    return render_template('bot.html', chat_history=chat_history)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.form['message']
    response = get_response(message)

    chat_history = session.get('chat_history', [])
    chat_history.append({"sender": "You", "text": message})
    chat_history.append({"sender": "Bot", "text": response})
    session['chat_history'] = chat_history

    audio_filename = f"response_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(audio_folder, audio_filename)

    try:
        tts = gTTS(text=response, lang='en')
        tts.save(audio_path)
    except Exception as e:
        return jsonify({"response": response, "audio_file": "", "error": str(e)})

    return jsonify({
        "response": response,
        "audio_file": f"/static/audio/{audio_filename}"
    })


r = sr.Recognizer()
mic = sr.Microphone()
lock = threading.Lock()

def speak1():
    with lock:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                text = r.recognize_google(audio)
                return text.lower()
        except Exception as e:
            print(f"Speech error: {e}")
            return None

@app.route('/speak', methods=['POST'])
def speak():
    speech = speak1()
    if not speech:
        return jsonify({
            "speech": "",
            "result": "Sorry, I couldn't understand the audio.",
            "audio_file": ""
        })

    result = get_response(speech)
    chat_history = session.get('chat_history', [])
    chat_history.append({"sender": "You", "text": speech})
    chat_history.append({"sender": "Bot", "text": result})
    session['chat_history'] = chat_history

    audio_filename = f"response_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(audio_folder, audio_filename)
    tts = gTTS(result)
    tts.save(audio_path)
    audio_url = f"/static/audio/{audio_filename}"

    return jsonify({
        "speech": speech,
        "result": result,
        "audio_file": audio_url
    })

if __name__ == '__main__':
    app.run(debug=False, port=3016)



