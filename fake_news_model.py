import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Add labels: 1 = Fake, 0 = True
df_fake["label"] = 1
df_true["label"] = 0

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Use article text as features
X = df["text"]
y = df["label"]

# Convert text → numeric features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print accuracy
print("Model Accuracy:", model.score(X_test, y_test))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
