import pandas as pd
import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("🔄 Loading datasets...")

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = 1   # Fake
real["label"] = 0   # Real

# Combine datasets
df = pd.concat([fake, real])

# Keep only required columns
df = df[["text", "label"]]

print("✅ Dataset loaded:", df.shape)

# Clean text
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

print("🔄 Cleaning text...")
df["text"] = df["text"].apply(clean)

# Convert text to numbers
print("🔄 Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
print("🔄 Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained & saved successfully!")
from sklearn.metrics import accuracy_score

pred = model.predict(X)
print("🎯 Model Accuracy:", accuracy_score(y, pred))