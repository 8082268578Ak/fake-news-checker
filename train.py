import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------
# 1. Load Dataset
# -------------------------
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = 0   # Fake
df_true["label"] = 1   # Real

df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# -------------------------
# 2. Clean Text
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# -------------------------
# 3. Train Test Split
# -------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. TF-IDF (IMPROVED)
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# 5. Model
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------
# 6. Accuracy
# -------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("🔥 Accuracy:", accuracy)

# -------------------------
# 7. Save Model
# -------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))