import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
#hhhhh
# Add labels
fake['label'] = 0
true['label'] = 1

# Combine
df = pd.concat([fake, true])#hhhhh

# Use only text + label
df = df[['text', 'label']]#hhhhh

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Training complete!")
