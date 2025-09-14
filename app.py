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
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return '''
        <h2> Fake News Detector</h2>
        <form method="POST">
         <textarea name="news" rows="5" cols="40" placeholder="Enter news text"></textarea><br>
         <input type="submit" value="Check News">
        </form>
        '''
    news_text=request.form.get('news', '').strip()
    if not news_text:
        return "Please enter news text"
    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)[0]
    result= "FAKE" if int(prediction) == 1 else "REAL"
    return f"<h3> Result: {result}</h3>"
if __name__ == '__main__':
    app.run(debug=True)
