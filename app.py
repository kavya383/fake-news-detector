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
@app.route('/')
def home():
    return "Flask server is running!"
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('news', '')
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return jsonify({"result": "FAKE" if int(prediction) == 1 else "REAL"})

if __name__ == '__main__':
    print("Starting local Flask server on localhost...")
    app.run(debug=True)
