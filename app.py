from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    if not text.strip():
        return render_template("index.html", result="⚠️ Enter some text")

    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0]

    confidence = round(max(prob) * 100, 2)

    if prediction == 1:
        result = "❌ Fake News"
        color = "fake"
    else:
        result = "✅ Real News"
        color = "real"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        color=color
    )

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)