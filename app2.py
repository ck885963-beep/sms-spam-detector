import streamlit as st
from flask import Flask, request, jsonify
import joblib
import sklearn

app = Flask(__name__)

# Load pre-trained spam detection model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return "SMS Spam Detector is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message field is required'}), 400

    vect_msg = vectorizer.transform([message])
    prediction = model.predict(vect_msg)

    st.title("Email/SMS Spam Classsifier")

    input_sms = st.text_area("Enter the message")
    if message:
        st.write("Prediction: SPAM or HAM here")

    return jsonify({'message': message, 'is_spam': bool(prediction[0])})