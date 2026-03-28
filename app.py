from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = max(prob) * 100

    # Category logic
    if confidence < 50:
        category = "fake"
    elif confidence < 80:
        category = "moderate"
    else:
        category = "real"

    return render_template("index.html",
                           prediction=pred,
                           confidence=round(confidence, 2),
                           category=category)

if __name__ == "__main__":
    app.run(debug=True)