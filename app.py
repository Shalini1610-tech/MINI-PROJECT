from flask import Flask, render_template, request, session, redirect
import pickle
import pandas as pd
import numpy as np
import joblib
import json
import speech_recognition as sr
from flask import jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import speech_recognition as sr
import nltk
from pydub import AudioSegment
import tempfile
import sqlite3

import os
from pydub import AudioSegment


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for sessions


# === Load Models and Preprocessors ===
with open('pcos_model.pkl', 'rb') as f:
    pcos_model = joblib.load(f)

with open('catboost_uti_model.pkl', 'rb') as f:
    uti_model = joblib.load(f)

with open('pcos_scaler.pkl', 'rb') as f:
    pcos_scaler = joblib.load(f)

with open('pcos_selector.pkl', 'rb') as f:
    pcos_selector = joblib.load(f)

# === Load Translations ===
def load_translations(lang):
    try:
        if lang == 'te':
            with open('/static/images/lang/te.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'en':
            with open('/main/static/images/lang/en.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'hi':
            with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/hi.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'pu':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/pu.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'bn':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/bn.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'od':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/od.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'as':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/as.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'gu':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/gu.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'ma':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/ma.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        elif lang == 'ta':
             with open('https://github.com/Shalini1610-tech/MINI-PROJECT/blob/main/static/images/lang/ta.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        
        else:
            # Default to English if unsupported language code is provided
            with open('https://raw.githubusercontent.com/Shalini1610-tech/MINI-PROJECT/main/static/images/lang/en.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading translations for language '{lang}': {str(e)}")
        return {}

# === Language Selection ===
@app.route('/set_language/<lang>')
def set_language(lang):
    session['language'] = lang
    return redirect(request.referrer or '/')

# === Routes ===
@app.route('/')
def home():
    lang = session.get('language', 'en')
    translations = load_translations(lang)
    return render_template('homepage2.html', translations=translations)

@app.route('/about')
def about():
    lang = session.get('language', 'en')
    translations = load_translations(lang)
    return render_template('about.html', translations=translations)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    lang = session.get('language', 'en')
    translations = load_translations(lang)
    submitted = False
    name = ""
    message = ""

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # Store into database
        conn = sqlite3.connect("contacts.db")
        c = conn.cursor()
        c.execute("INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)", (name, email, message))
        conn.commit()
        conn.close()

        submitted = True

    return render_template("contact.html", submitted=submitted, name=name, message=message, translations=translations)



@app.route('/diagnosis')
def diagnosis():
    lang = session.get('language', 'en')
    translations = load_translations(lang)
    return render_template('diagnosis.html' , translations=translations)


# === PCOS Diagnosis ===
@app.route('/pcosm', methods=['GET', 'POST'])
def pcosm():
    result = None
    lang = session.get('language', 'en')
    translations = load_translations(lang)

    if request.method == 'POST':
        try:
            input_features = [
                float(request.form['Age (yrs)']),
                float(request.form['Weight (Kg)']),
                float(request.form['Height(Cm)']),
                float(request.form['BMI']),
                float(request.form['Blood Group']),
                float(request.form['Pulse rate(bpm)']),
                float(request.form['RR (breaths/min)']),
                float(request.form['Hb(g/dl)']),
                1 if request.form['Cycle(R/I)'] == 'R' else 0,
                float(request.form['Cycle length(days)']),
                float(request.form['Marraige Status (Yrs)']),
                1 if request.form['Pregnant(Y/N)'].lower() == 'yes' else 0,
                float(request.form['No. of aborptions']),
                float(request.form['FSH(mIU/mL)']),
                float(request.form['LH(mIU/mL)']),
                float(request.form['FSH/LH']),
                float(request.form['Hip(inch)']),
                float(request.form['Waist(inch)']),
                float(request.form['Waist:Hip Ratio']),
                float(request.form['TSH (mIU/L)']),
                float(request.form['AMH(ng/mL)']),
                float(request.form['PRL(ng/mL)']),
                float(request.form['Vit D3 (ng/mL)']),
                float(request.form['PRG(ng/mL)']),
                float(request.form['RBS(mg/dl)']),
                1 if request.form['Weight gain(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['hair growth(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Skin darkening (Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Hair loss(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Pimples(Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Fast food (Y/N)'].lower() == 'yes' else 0,
                1 if request.form['Reg.Exercise(Y/N)'].lower() == 'yes' else 0,
                float(request.form['BP _Systolic (mmHg)']),
                float(request.form['BP _Diastolic (mmHg)']),
                float(request.form['Follicle No. (L)']),
                float(request.form['Follicle No. (R)']),
                float(request.form['Avg. F size (L) (mm)']),
                float(request.form['Avg. F size (R) (mm)']),
                float(request.form['Endometrium (mm)'])
            ]

            input_array = np.array([input_features])
            scaled_input = pcos_scaler.transform(input_array)
            selected_input = pcos_selector.transform(scaled_input)
            prediction = pcos_model.predict(selected_input)
            result = translations['pcos_result_positive'] if prediction[0] == 1 else translations['pcos_result_negative']

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('pcosm.html', result=result, translations=translations)

@app.route('/speech_diagnosis', methods=['POST'])
def speech_diagnosis():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎤 Listening for user symptoms...")
            audio = recognizer.listen(source)

        transcript = recognizer.recognize_google(audio)
        print("User said:", transcript)

        stop_words = set(stopwords.words('english'))
        cleaned_input = ' '.join([w for w in word_tokenize(transcript.lower()) if w.isalnum() and w not in stop_words])

        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([cleaned_input] + stored_symptoms)
        similarities = cosine_similarity(vectors[0:1], vectors[1:])
        best_match_index = similarities.argmax()
        diagnosis = stored_diagnoses[best_match_index]

        return jsonify({"transcript": transcript, "diagnosis": diagnosis})
    except Exception as e:
        print(" Error in speech diagnosis:", e)
        return jsonify({"transcript": "", "diagnosis": "Speech recognition error"})




@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    try:
        # Save uploaded blob
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        audio_file.save(temp_input.name)

        # Convert to wav
        sound = AudioSegment.from_file(temp_input.name)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export("processed.wav", format="wav")

        # Recognize
        with sr.AudioFile("processed.wav") as source:
            audio = recognizer.record(source)
            transcript = recognizer.recognize_google(audio)

        # Dummy diagnosis logic
        symptoms = transcript.lower()
        if "burning" in symptoms or "urine" in symptoms:
            diagnosis = "UTI"
        elif "irregular" in symptoms or "periods" in symptoms or "hormonal imbalance" in symptoms or "acne" in symptoms or "excess " in symptoms:
            diagnosis = "PCOS"
        else:
            diagnosis = "Normal"

        return jsonify({"transcript": transcript, "diagnosis": diagnosis})
    
    except Exception as e:
        print("❌ Speech recognition error:", e)
        return jsonify({"transcript": "", "diagnosis": "Could not recognize speech"})


# === UTI Diagnosis ===
@app.route('/uti', methods=['GET', 'POST'])
def uti():
    result = None
    lang = session.get('language', 'en')
    translations = load_translations(lang)

    if request.method == 'POST':
        try:
            data = request.form
            values = [
                float(data.get('Temperature of patient', 0.0)),
                1 if data.get('Occurrence of nausea') == 'yes' else 0,
                1 if data.get('Lumbar pain') == 'yes' else 0,
                1 if data.get('Urine pushing (continuous need for urination)') == 'yes' else 0,
                1 if data.get('Micturition pains') == 'yes' else 0,
                1 if data.get('Burning of urethra, itch, swelling of urethra outlet') == 'yes' else 0,
                1 if data.get('Inflammation of urinary bladder') == 'yes' else 0
            ]

            input_df = pd.DataFrame([values], columns=[
                'Temperature of patient',
                'Occurrence of nausea',
                'Lumbar pain',
                'Urine pushing (continuous need for urination)',
                'Micturition pains',
                'Burning of urethra, itch, swelling of urethra outlet',
                'Inflammation of urinary bladder'
            ])

            prediction = uti_model.predict(input_df)[0]
            result = translations.get('uti_result_positive', 'Positive for UTI') if prediction == 1 else translations.get('uti_result_negative', 'Negative for UTI')

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('uti.html', result=result, translations=translations)
stored_symptoms = [
    "burning while urinating and fishy smell in urine",
    "frequent urination with discomfort",
    "irregular periods and excess facial hair",
    "acne and weight gain",
    "no symptoms"
]
stored_diagnoses = [
    "UTI",
    "UTI",
    "PCOS",
    "PCOS",
    "Normal"
]





# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
