# test_microphone_diagnosis.py

import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download required data once
#nltk.download('punkt')
#nltk.download('stopwords')

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

recognizer = sr.Recognizer()
with sr.Microphone(device_index=1) as source:

    print("🎤 Speak your symptoms...")
    audio = recognizer.listen(source)

try:
    transcript = recognizer.recognize_google(audio)
    print("You said:", transcript)

    stop_words = set(stopwords.words('english'))
    cleaned_input = ' '.join([w for w in word_tokenize(transcript.lower()) if w.isalnum() and w not in stop_words])

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([cleaned_input] + stored_symptoms)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])
    best_match_index = similarities.argmax()
    diagnosis = stored_diagnoses[best_match_index]

    print("Predicted diagnosis:", diagnosis)

except Exception as e:
    print("❌ Error:", e)
