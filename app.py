from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
vec_path = os.path.join(base_dir, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vec_path, "rb"))
@app.route('/', methods=['GET','POST])
def home():
    return "Flask server is running!"
def index():
    if request.method == 'GET':
        return "Fake News Detector API is running! Send POST request with JSON {'news':'text'}"
    data = request.get_json()
    text = data.get('news', '')
    if not text:
        return jsonify({"error":"No news text provided"}),400
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return jsonify({"result": "FAKE" if int(prediction) == 1 else "REAL"})

if __name__ == '__main__':
    app.run(debug=True)
