from flask import Flask, render_template, request, redirect, url_for
import pickle
import json
from fact_check import google_fact_check

app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Store history
history = []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    # ML prediction
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob) * 100

    # Decision logic
    if pred == 0:
        category = "fake" if confidence > 80 else "moderate"
    else:
        category = "real" if confidence > 80 else "moderate"

    # 🔍 Google Fact Check
    fact_status, fact_results = google_fact_check(news)

    # Save history
    history.append({
        "text": news,
        "category": category,
        "confidence": round(confidence, 2)
    })

    return redirect(url_for(
        "result",
        category=category,
        confidence=round(confidence, 2),
        text=news,
        fact_status=fact_status,
        fact_results=json.dumps(fact_results)
    ))


@app.route("/result")
def result():
    category = request.args.get("category")
    confidence = request.args.get("confidence")
    text = request.args.get("text")
    fact_status = request.args.get("fact_status")

    fact_results = request.args.get("fact_results")

    if fact_results:
        fact_results = json.loads(fact_results)
    else:
        fact_results = []

    return render_template(
        "result.html",
        category=category,
        confidence=confidence,
        text=text,
        fact_status=fact_status,
        fact_results=fact_results
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", history=history)


if __name__ == "__main__":
    app.run(debug=True)