from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")#hhhhh
#ghh
@app.route('/predict', methods=['POST'])#yyyy
def predict():
    try:
        news = request.form['news']

        vec = vectorizer.transform([news])
        prob = model.predict_proba(vec)[0][1]

        confidence = round(prob * 100, 2)

        # 3-class logic
        if prob < 0.4:
            result = "FAKE NEWS ❌"#hhhhh
        elif prob < 0.7:
            result = "MODERATE NEWS ⚠"
        else:
            result = "TRUE NEWS ✅"

        return render_template(
            "result.html",
            prediction=result,
            text=news,
            confidence=confidence
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
