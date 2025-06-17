from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
from gensim.models import Word2Vec
import os
import threading

app = Flask(__name__)

base_dir = os.getcwd()
svm_model_path = os.path.join(base_dir, "svm_model.pkl")
w2v_model_path = os.path.join(base_dir, "word2vec.model")

svm_model = joblib.load(svm_model_path)
w2v_model = Word2Vec.load(w2v_model_path)

def sentence_to_vec(sentence, model):
    words = sentence.lower().split()
    vec = np.zeros(model.vector_size)
    count = 0
    for w in words:
        if w in model.wv:
            vec += model.wv[w]
            count += 1
    if count != 0:
        vec /= count
    return vec

# Route cho trang home (home.html)
@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

# Route cho trang index (index.html) - phân tích review
@app.route("/index", methods=["GET", "POST"])
def index():
    prediction_text = ""
    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            X_vec = sentence_to_vec(review, w2v_model).reshape(1, -1)
            prediction = svm_model.predict(X_vec)[0]
            prediction_text = f"Kết quả dự đoán: {prediction}"
        else:
            prediction_text = "Vui lòng nhập review hợp lệ."
    return render_template("index.html", prediction_text=prediction_text)

# Route chuyển hướng từ home.html đến index.html khi bấm "Start Analysis"
@app.route("/", methods=["GET"])
def start_analysis():
    return redirect(url_for('home'))

def run_app():
    app.run(debug=True, use_reloader=False)

# Khởi chạy Flask trên một thread
thread = threading.Thread(target=run_app)
thread.start()

