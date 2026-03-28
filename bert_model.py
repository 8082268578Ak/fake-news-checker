from transformers import pipeline

# Load pretrained BERT model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_news(text):
    result = classifier(text)[0]

    label = result['label']
    score = result['score']

    # Convert to percentage
    confidence = score * 100

    # Decision logic
    if confidence < 60:
        return "moderate", confidence

    if label == "POSITIVE":
        return "real", confidence
    else:
        return "fake", confidence