from flask import Flask, render_template, request, redirect, url_for
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

    # Transform input
    vec = vectorizer.transform([news])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = max(prob) * 100

    # 🔥 Improved logic
    if pred == 0:
        if confidence > 80:
            category = "fake"
        else:
            category = "moderate"
    else:
        if confidence > 80:
            category = "real"
        else:
            category = "moderate"

    return redirect(url_for(
        "result",
        category=category,
        confidence=round(confidence, 2),
        text=news
    ))


@app.route("/result")
def result():
    category = request.args.get("category")
    confidence = request.args.get("confidence")
    text = request.args.get("text")

    return render_template(
        "result.html",
        category=category,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)